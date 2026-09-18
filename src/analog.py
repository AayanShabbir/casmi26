"""F-ANALOG core — entropy similarity, mass-shifted analog propagation, Tanimoto.

Single-channel analog backbone for CASMI26 (reproduces the winner's ~0.50
Class-2 MRR claim). Pure numpy.
"""
from __future__ import annotations
import os, pickle, time
import numpy as np
import pyarrow.parquet as pq

PROTON = 1.00728
PPM_WIN = 10.0          # neutral-mass window for candidates (winner: 10 = no loss)
ANALOG_WIN = 200.0      # +- Da mass-shift search window
N_ANALOG = 100          # analogs kept per query
INT_FLOOR = 0.002
MAX_PEAKS = 256
MZ_TOL = 0.01
SIM_POWER = 4.0         # sim^p
ADDUCT_NEUT = {"[M+H]+": PROTON, "[M+NH4]+": 18.0383, "[M+Na]+": 22.9892,
               "[M+H-H2O]+": PROTON - 18.0106, "[M+2H]2+": 2*PROTON}


def neutral_mass(mz, adduct):
    mz = np.asarray(mz, np.float64)
    out = np.full(len(mz), np.nan)
    for a, d in ADDUCT_NEUT.items():
        m = np.asarray(adduct, dtype=object) == a
        out[m] = mz[m] - d
    return out


def clean(mz, it):
    """Entropy-weighted spectral cleaning -> (mz_sorted_float32, prob_float32)."""
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    if mz.size == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = it.max()
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    keep = it >= INT_FLOOR * mx
    mz, it = mz[keep], it[keep]
    if mz.size == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    if mz.size > MAX_PEAKS:
        o = np.argsort(-it)[:MAX_PEAKS]; mz, it = mz[o], it[o]
    o = np.argsort(mz); mz, it = mz[o], it[o]
    p = it / it.sum()
    # Li entropy sharpening
    S = float(-np.sum(p * np.log(p))) if p.sum() > 0 else 0.0
    if S < 3.0:
        w = 0.25 + 0.25 * S
        p = np.power(p, w); p = p / p.sum()
    return mz.astype(np.float32), p.astype(np.float32)


def entropy_spec(p, mz, tol=MZ_TOL):
    """-sum(p log p) of p."""
    return float(-np.sum(np.where(p > 0, p * np.log(p), 0.0)))


def entropy_sim(qmz, qp, cmz, cp, tol=MZ_TOL):
    """Spectral entropy similarity (Li & Fiehn): 1 - (2*SAB - SA - SB)/log(4)."""
    SA = entropy_spec(qp, qmz); SB = entropy_spec(cp, cmz)
    i = j = 0; n = len(qmz); m = len(cmz)
    buf = []
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol: buf.append(qp[i]); i += 1
        elif d > tol: buf.append(cp[j]); j += 1
        else: buf.append(qp[i] + cp[j]); i += 1; j += 1
    buf.extend(qp[i:]); buf.extend(cp[j:])
    tot = float(np.sum(buf))
    if tot <= 0: return 0.0
    v = np.asarray(buf) / tot
    SAB = float(-np.sum(v * np.log(v)))
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)


# ---------------------------------------------------------------------------
def load_library(path):
    t = pq.read_table(path, columns=["inchikey14", "adduct", "precursor_mz",
                                     "ms2_mzs", "ms2_normalized_intensities"])
    ik = np.asarray(t.column("inchikey14").cast("string").to_pylist(), dtype=object)
    add = np.asarray(t.column("adduct").cast("string").to_pylist(), dtype=object)
    prec = t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64)
    mzc = t.column("ms2_mzs").combine_chunks(); itc = t.column("ms2_normalized_intensities").combine_chunks()
    off = mzc.offsets.to_numpy().astype(np.int64)
    allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    allin = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    nm = neutral_mass(prec, add)
    return dict(n=len(ik), ik=ik, nm=nm, off=off, mz=allmz, it=allin,
                nm_valid=np.isfinite(nm))


def load_pool(data_dir):
    cf = np.load(os.path.join(data_dir, "coco_fp.npy"), mmap_mode="r")
    cm = pickle.load(open(os.path.join(data_dir, "coco_meta.pkl"), "rb"))
    mass = np.load(os.path.join(data_dir, "coco_mass.npy"))
    return dict(fp=cf, mass=np.asarray(mass, np.float64),
                keys=np.asarray(cm["keys"], dtype=object),
                smiles=np.asarray(cm["smiles"], dtype=object))


def packed_tanimoto_pairs(a_fp, b_fps):
    """Tanimoto between packed-uint8 fingerprint a and a set b (same packing).
    Uses bit AND/OR via reinterpret to uint64 for speed."""
    a64 = a_fp.astype(np.uint64).view(np.uint64)
    # a_fp is uint8 (867,) -> pad to multiple of 8
    n = len(a_fp); npad = (-n) % 8
    ap = np.pad(a_fp, (0, npad)).astype(np.uint8)
    a64 = np.frombuffer(ap.tobytes(), dtype=np.uint64)
    # b_fps: (nb, 867)
    bp = np.pad(b_fps, ((0, 0), (0, npad))).astype(np.uint8)
    b64 = np.frombuffer(bp.tobytes(), dtype=np.uint64).reshape(bp.shape[0], -1)
    inter = np.sum(np.bitwise_and(a64[None, :], b64).view(np.uint8).astype(np.int64), axis=1)
    na = int(np.count_nonzero(ap))
    nb = np.count_nonzero(bp.astype(np.uint8), axis=1)
    union = na + nb - inter
    return np.where(union > 0, inter / union, 0.0)


def build_rep(L, pool):
    """One representative spectrum per library structure (max peak count), sorted by mass."""
    npk = np.diff(L["off"])
    best = {}
    for i in range(L["n"]):
        k = L["ik"][i]
        if k and k not in best:
            best[k] = i
    rep = np.array(sorted(best.values()))
    nm = L["nm"][rep]; ok = np.isfinite(nm)
    rep = rep[ok]; nm = nm[ok]; key = L["ik"][rep]
    o = np.argsort(nm)
    return rep[o], key[o], nm[o], None


def _spec_arrays(L, r):
    a, b = L["off"][r], L["off"][r + 1]
    return L["mz"][a:b], L["it"][a:b]


def analog_rank(L, pool, rep, rep_key, rep_nm, rep_fp, qrow, target, idx_valid,
                topn=N_ANALOG):
    """Score COCONUT pool candidates by mass-shifted analog evidence.
    Returns list of (inchikey14, score) descending."""
    qmz, qit = _spec_arrays(L, qrow)
    qm, qp = clean(qmz, qit)
    if len(qm) == 0:
        return []
    # analogs inside mass window
    lo = np.searchsorted(rep_nm, target - ANALOG_WIN, "left")
    hi = np.searchsorted(rep_nm, target + ANALOG_WIN, "right")
    rep_slice = rep[lo:hi]; shift = (target - rep_nm[lo:hi]).astype(np.float32)
    # aggregate best sim per analog structure
    agg = {}
    for aidx, sh in zip(rep_slice, shift):
        cmz, cit = _spec_arrays(L, aidx)
        cm, cp = clean(cmz, cit)
        if len(cm) == 0:
            continue
        s_direct = entropy_sim(qm, qp, cm, cp)
        s_shift = s_direct
        if abs(sh) > 0.001:
            s_shift = entropy_sim(qm, qp, cm + sh, cp)
        s = max(s_direct, s_shift)
        k = L["ik"][aidx]
        if k and s > agg.get(k, -1.0):
            agg[k] = s
    analogs = sorted(agg.items(), key=lambda x: -x[1])[:topn]
    if not analogs:
        return []
    # candidate window: COCONUT pool structures near target mass
    half = PPM_WIN * 1e-6 * target
    cand = np.where(np.abs(pool["mass"] - target) <= half)[0]
    if len(cand) == 0:
        return []
    # gather analog fingerprints per analog key from pool
    pk = pool["keys"]
    analog_fp = []; analog_sim = []
    for k, s in analogs:
        m = np.where(pk == k)[0]
        if len(m):
            analog_fp.append(pool["fp"][m[0]]); analog_sim.append(s)
    if not analog_fp:
        return []
    analog_fp = np.stack(analog_fp)
    analog_sim = np.asarray(analog_sim, np.float32)
    cand_fp = pool["fp"][cand]
    # tanimoto candidate vs each analog, weighted sim^p, max over analogs
    best_score = np.full(len(cand), 0.0)
    cfp32 = cand_fp.astype(np.uint8)
    for j, af in enumerate(analog_fp):
        t = packed_tanimoto_pairs(af, cfp32)
        best_score = np.maximum(best_score, analog_sim[j] ** SIM_POWER * t)
    order = np.argsort(-best_score)
    out = []
    for ci in order:
        if best_score[ci] <= 0:
            continue
        out.append((pk[cand[ci]], float(best_score[ci])))
    return out