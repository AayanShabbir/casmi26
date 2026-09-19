"""F-CLASS — class-prior calibration of the HistGB ranker proba (CASMI26 P2).

The Phase-1 GBDT is trained with synthetic 50/50 class weighting, so its
predict_proba is inflated relative to the true empirical prior (112 positives /
2163 training rows). F-CLASS recalibrates per candidate *conditioned on which
class it belongs to* (Class-1 = direct library-match candidate, lib_sim>0;
Class-2 = analog-only candidate). A per-class quantile prior is monotone within
a class but corrects the relative standing *across* classes, so it is a genuine
ordering correction (a global monotone transform would be an MRR no-op).

Two layers, both monotone and reproducible:
  1. class-prior likelihood weighting: score = proba / prior_c
  2. optional per-class isotonic recalibration fit on held-out train queries.

The public entry point is calibrate(proba, y_prior) which returns calibrated
scores; classes and calib make it class-conditional when the runner supplies
the per-candidate class vector.
"""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression


# f_ranker feature columns: index 19 is the library-match flag (lib_sim>0).
LIB_FLAG_COL = 19


def class_of(Xf):
    """Per-candidate class from the F-RANKER feature matrix.
    Class-1 = direct library match (lib_flag > 0), Class-2 = analog-only.
    Returns an int array len == Xf.shape[0] with values in {1, 2}.
    """
    if Xf.ndim != 2 or Xf.shape[1] <= LIB_FLAG_COL:
        raise ValueError("f_class: expected (n_cand, NFEAT) feature matrix")
    return np.where(Xf[:, LIB_FLAG_COL] > 0, 1, 2).astype(np.int64)


def empirical_priors(proba_rows, y_rows, class_ids):
    """Per-class empirical positive ratio (the 112/2163-style prior).
    Inputs are concatenated row arrays over many train queries.
    Returns dict {1: p1, 2: p2}.
    """
    priors = {}
    for c in (1, 2):
        m = class_ids == c
        pos = int(np.asarray(y_rows)[m].sum())
        n = int(m.sum())
        priors[c] = pos / max(1, n)
    return priors


def fit_recalibration(proba_rows, y_rows, class_ids):
    """Fit per-class isotonic recalibration on held-out train queries.

    Returns (prior_dict, calib_dict):
      prior_dict : {c: empirical positive prior}
      calib_dict : {c: IsotonicRegression fitted on proba->y for class c}
    Isotonic is monotone non-decreasing, so it never inverts intra-class order;
    combined with the class prior it re-orders across classes.
    """
    proba_rows = np.asarray(proba_rows, np.float64)
    y_rows = np.asarray(y_rows)
    class_ids = np.asarray(class_ids)
    priors = empirical_priors(proba_rows, y_rows, class_ids)
    calib = {}
    for c in (1, 2):
        m = class_ids == c
        p = proba_rows[m]
        y = y_rows[m]
        if len(p) < 4 or len(np.unique(y)) < 2:
            calib[c] = None  # degenerate class: rely on prior weighting only
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p, y.astype(np.float64))
        calib[c] = iso
    return priors, calib


def calibrate(proba, y_prior, classes=None, calib=None):
    """Return recalibrated scores.

    proba     : float array of model predicted probabilities (len n).
    y_prior   : empirical positive prior. A scalar applies globally; a dict
                {1: p1, 2: p2} applies per-class when `classes` is given.
    classes   : optional int array len n in {1,2}; if None all treated class-1.
    calib     : optional {c: IsotonicRegression} from fit_recalibration.

    The calibrated score order is what ranking consumes; a constant global
    scaling is irrelevant to MRR, only the class-conditional re-ordering and
    the monotone isotonic correction change the outcome.
    """
    proba = np.asarray(proba, np.float64)
    p = np.clip(proba, 1e-9, 1.0 - 1e-9)
    n = p.shape[0]
    if classes is None:
        classes = np.ones(n, np.int64)
    classes = np.asarray(classes, np.int64)
    # raw ranker probability first
    out = p.copy()
    if calib:
        for c in (1, 2):
            iso = calib.get(c)
            if iso is None:
                continue
            m = classes == c
            if m.any():
                pred = np.where(np.isfinite(iso.predict(p[m])), iso.predict(p[m]), p[m])
                out[m] = pred
    # class-prior likelihood weighting: divide by the class's empirical prior
    if isinstance(y_prior, dict):
        for c in (1, 2):
            pr = y_prior.get(c, 0.05)
            if pr > 0:
                m = classes == c
                out[m] = out[m] / pr
    else:
        pr = float(y_prior)
        if pr > 0:
            out = out / pr
    return out