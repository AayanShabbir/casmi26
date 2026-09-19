"""F-ANALOG accelerated core — precomputed cleaned spectra + njit entropy loops."""
from __future__ import annotations
import numpy as np
import pyarrow.parquet as pq
from numba import njit

PROTON = 1.00728
INT_FLOOR = 0.002
MAX_PEAKS = 256
MZ_TOL = 0.01
ADDUCT_NEUT = {"[M+H]+": PROTON, "[M+NH4]+": 18.0383, "[M+Na]+": 22.9892,
               "[M+H-H2O]+": PROTON - 18.0106, "[M+2H]2+": 2*PROTON,
               # test-set adducts missed by the original positive-only map (would
               # NaN neutral mass -> molecules dropped from submission)
               "[M-H]-": -PROTON,
               "[M+CH2O2-H]-": -(46.0255 - PROTON),   # CH2O2 mass minus H
               "[M+K]+": 38.9637,
               "[M+Cl]-": -34.9693}


def neutral_mass(mz, adduct):
    mz = np.asarray(mz, np.float64)
    out = np.full(len(mz), np.nan)
    for a, d in ADDUCT_NEUT.items():
        m = np.asarray(adduct, dtype=object) == a
        out[m] = mz[m] - d
    return out


@njit(cache=True, fastmath=True)
def _clean(mz, it, floor, topk):
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx: mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    idx = np.empty(n, np.int64); c = 0
    for i in range(n):
        if it[i] >= thr: idx[c] = i; c += 1
    if c == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c): v[i] = it[idx[i]]
        o = np.argsort(v)
        keep = o[c - topk:]
        idx2 = np.empty(topk, np.int64)
        for j in range(topk): idx2[j] = idx[o[c - topk + j]]
        idx = idx2; c = topk
    om = np.empty(c, np.float32); oi = np.empty(c, np.float32)
    mz2 = np.empty(c, np.float64); ii = np.empty(c, np.float64)
    for j in range(c):
        mz2[j] = mz[idx[j]]; ii[j] = it[idx[j]]
    o = np.argsort(mz2)
    s = 0.0
    for j in range(c):
        om[j] = mz2[o[j]]; oi[j] = ii[o[j]]; s += ii[o[j]]
    if s <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    p = np.empty(c, np.float64)
    for j in range(c): p[j] = ii[o[j]] / s
    # entropy weighting (Li 2021)
    S = 0.0
    for j in range(c):
        if p[j] > 0: S -= p[j] * np.log(p[j])
    if S < 3.0:
        w = 0.25 + 0.25 * S
        s2 = 0.0
        for j in range(c): p[j] = p[j] ** w; s2 += p[j]
        if s2 > 0:
            for j in range(c): p[j] /= s2
    return om.astype(np.float32), p.astype(np.float32)


@njit(cache=True, fastmath=True)
def _entropy_sim(qmz, qp, cmz, cp, tol):
    n = len(qmz); m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0: SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0: SB -= cp[x] * np.log(cp[x])
    buf = np.empty(n + m, np.float64); b = 0
    i = 0; j = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol: buf[b] = qp[i]; i += 1; b += 1
        elif d > tol: buf[b] = cp[j]; j += 1; b += 1
        else: buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1
    while i < n: buf[b] = qp[i]; i += 1; b += 1
    while j < m: buf[b] = cp[j]; j += 1; b += 1
    tot = 0.0
    for x in range(b): tot += buf[x]
    if tot <= 0: return 0.0
    SAB = 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0: SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)


@njit(cache=True, fastmath=True)
def _mass_shift_sim(qmz, qp, cmz, cp, tol, shift):
    # direct vs shifted (shift all cmz by `shift`)
    a = _entropy_sim(qmz, qp, cmz, cp, tol)
    if -0.001 < shift < 0.001:
        return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)): sm[i] = cmz[i] + shift
    b = _entropy_sim(qmz, qp, sm, cp, tol)
    return a if a > b else b


@njit(cache=True, parallel=True, fastmath=True)
def _search_shift_batch(qmz, qp, rep_rows, coff, cmz, cp, tol, shifts):
    """Score every rep row in one parallel njit call (winner's search_shift).
    rep_rows: int array of library row indices; returns float array of sims."""
    out = np.empty(len(rep_rows), np.float32)
    for k in range(len(rep_rows)):   # prange would skip reps, keep serial-safe
        r = rep_rows[k]
        a, b = coff[r], coff[r + 1]
        if b <= a:
            out[k] = 0.0
            continue
        out[k] = _mass_shift_sim(qmz, qp, cmz[a:b], cp[a:b], tol, shifts[k])
    return out


class Lib:
    def __init__(self, path):
        t = pq.read_table(path, columns=["inchikey14", "adduct", "precursor_mz",
                                         "ms2_mzs", "ms2_normalized_intensities"])
        self.ik = np.asarray(t.column("inchikey14").cast("string").to_pylist(), dtype=object)
        add = np.asarray(t.column("adduct").cast("string").to_pylist(), dtype=object)
        prec = t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64)
        self.nm = neutral_mass(prec, add)
        mzc = t.column("ms2_mzs").combine_chunks()
        itc = t.column("ms2_normalized_intensities").combine_chunks()
        self.off = mzc.offsets.to_numpy().astype(np.int64)
        self.mz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
        self.it = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
        self.n = len(self.ik)
        self._clean_all()

    def _clean_all(self):
        # two-pass: count then fill (no concat-in-loop O(n^2))
        off = np.empty(self.n + 1, np.int64); off[0] = 0
        for r in range(self.n):
            a, b = self.off[r], self.off[r + 1]
            cm, cp = _clean(self.mz[a:b].astype(np.float32),
                            self.it[a:b].astype(np.float32), INT_FLOOR, MAX_PEAKS)
            off[r + 1] = off[r] + len(cm)
        total = int(off[-1])
        cmzs = np.empty(total, np.float32); cps = np.empty(total, np.float32)
        for r in range(self.n):
            a, b = self.off[r], self.off[r + 1]
            cm, cp = _clean(self.mz[a:b].astype(np.float32),
                            self.it[a:b].astype(np.float32), INT_FLOOR, MAX_PEAKS)
            s, e = off[r], off[r + 1]
            cmzs[s:e] = cm; cps[s:e] = cp
        self.cmz = cmzs; self.cp = cps; self.coff = off