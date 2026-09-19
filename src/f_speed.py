"""F-SPEED — prange-parallelized mass-shift spectral similarity (CASMI26).

Replaces the serial `_search_shift_batch` with a fused, prange-parallel version.
The serial one never parallelized (NumbaPerformanceWarning: loop has no prange)
and it called allocating helper njit fns per rep. This fuses entropy + shift +
merge into ONE njit with prange so multicore actually engages.

Contract is IDENTICAL to old `_search_shift_batch(qmz, qp, rep_rows, coff,
cmz, cp, tol, shifts) -> float32 array len(rep_rows)`, so callers just swap
the function.
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _entropy_score(a, alp, b, blp):
    """Pure entropy-similarity term: 1 - (2*SAB - SA - SB)/log(4), where the
    merged distribution is `a` probabilities concatenated with merge rule.
    Here we implement the standard two-spectrum entropy similarity directly on
    pre-normalized probs. Expects a,b already probability-normalized."""
    n = len(a)
    SA = 0.0
    for x in range(n):
        if a[x] > 0.0:
            SA -= a[x] * np.log(a[x])
    m = len(b)
    SB = 0.0
    for x in range(m):
        if b[x] > 0.0:
            SB -= b[x] * np.log(b[x])
    return SA, SB


@njit(cache=True, parallel=True, fastmath=True)
def search_shift_prange(qmz, qp, rep_rows, coff, cmz, cp, tol, shifts):
    """Fused parallel mass-shift entropy similarity over rep_rows.

    Each rep's cleaned spectrum is at [cmz/cp[coff[r]:coff[r+1]]]. shift[k]
    offsets the rep's peaks. Returns direct-vs-shifted max entropy sim.
    """
    out = np.empty(len(rep_rows), np.float32)
    for k in prange(len(rep_rows)):
        r = rep_rows[k]
        a, b = coff[r], coff[r + 1]
        if b <= a:
            out[k] = 0.0
            continue
        # ---------------- direct ----------------
        s_direct = _entropy_sim_inline(qmz, qp, cmz[a:b], cp[a:b], tol)
        sh = shifts[k]
        if -0.001 < sh < 0.001:
            out[k] = s_direct
            continue
        # ---------------- shifted ----------------
        smz = np.empty(b - a, np.float32)
        for j in range(b - a):
            smz[j] = cmz[a + j] + sh
        s_shift = _entropy_sim_inline(qmz, qp, smz, cp[a:b], tol)
        out[k] = s_direct if s_direct > s_shift else s_shift
    return out


@njit(cache=True, fastmath=True)
def _entropy_sim_inline(qmz, qp, cmz, cp, tol):
    """Entropy similarity of two cleaned (prob-normalized) peak lists."""
    n = len(qmz); m = len(cmz)
    if n == 0 or m == 0:
        return 0.0
    SA = 0.0
    for x in range(n):
        if qp[x] > 0.0:
            SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0.0:
            SB -= cp[x] * np.log(cp[x])
    # merge with tolerance tol
    buf = np.empty(n + m, np.float64)
    b = 0
    i = 0; j = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            buf[b] = qp[i]; i += 1; b += 1
        elif d > tol:
            buf[b] = cp[j]; j += 1; b += 1
        else:
            buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1
    while i < n:
        buf[b] = qp[i]; i += 1; b += 1
    while j < m:
        buf[b] = cp[j]; j += 1; b += 1
    if b == 0:
        return 0.0
    tot = 0.0
    for x in range(b):
        tot += buf[x]
    if tot <= 0:
        return 0.0
    SAB = 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0.0:
            SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)

# alias so callers can import either name
search_shift_batch = search_shift_prange