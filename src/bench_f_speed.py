"""F-SPEED benchmark v2 — pick a PEAK-RICH query + real reps (valid similarity cost).

The v1 bench accidentally used a query with q_peaks=0 (empty cleaned spectrum),
so all sims short-circuited and both functions were ~0ms — meaningless. This
version only accepts a query with >=30 peaks and reps that carry peaks, and it
explicitly enables 8 numba threads, then verifies equivalence on the real kernel.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import analog_accel as A
import f_speed as F

try:
    import numba
    from numba import config as nbconfig
    print(f"[numba] threads={nbconfig.NUMBA_NUM_THREADS}", flush=True)
    try:
        numba.set_num_threads(8)
        print(f"[numba] set_num_threads(8) -> {numba.get_num_threads()}", flush=True)
    except Exception as e:
        print(f"[numba] set_num_threads n/a: {e}", flush=True)
except Exception as e:
    print(f"[numba] n/a: {e}", flush=True)

TRAIN = os.path.join(ROOT, "train.parquet")
N_REPS = 3000
ANALOG_WIN = 200.0
MIN_Q_PEAKS = 30


def main():
    t0 = time.time()
    L = A.Lib(TRAIN)
    print(f"[lib+clean] {L.n} spectra cleaned {time.time()-t0:.0f}s", flush=True)

    # rep rows sorted by neutral mass (mirror analog_run_v2)
    best = {}
    for r in range(L.n):
        k = L.ik[r]
        if k and np.isfinite(L.nm[r]) and k not in best:
            best[k] = r
    rep = np.array(sorted(best.values()))
    nmr = L.nm[rep]
    oor = np.argsort(nmr)
    rep = rep[oor]; nmr = nmr[oor]
    print(f"[rep] {len(rep)} representatives", flush=True)

    # candidate reps that have a cleaned spectrum w/ >=MIN_Q_PEAKS peaks
    peaky = []
    for r in rep:
        a, b = L.off[r], L.off[r + 1]
        if b > a and len(L.cmz[a:b]) >= MIN_Q_PEAKS:
            peaky.append(r)
    print(f"[peaky] {len(peaky)} reps with >= {MIN_Q_PEAKS} cleaned peaks", flush=True)

    # pick a peaky query whose +-200 window covers >= N_REPS reps
    chosen = None
    for r in peaky:
        target = np.float64(L.nm[r])
        lo = np.searchsorted(nmr, target - ANALOG_WIN, "left")
        hi = np.searchsorted(nmr, target + ANALOG_WIN, "right")
        if hi - lo >= N_REPS:
            chosen = (r, target, lo, hi)
            break

    if chosen is None:
        # no single query covers 3000 in +-200: widen to nearest 3000 reps around
        # the FIRST peaky query's mass index (still real cross-mass similarity)
        r0 = peaky[0]
        target = np.float64(L.nm[r0])
        pos = int(np.searchsorted(nmr, target, "left"))
        lo = max(0, pos - N_REPS // 2)
        hi = min(len(nmr), pos + N_REPS - (N_REPS // 2))
        r = r0
        chosen = (r, target, lo, hi)
        print(f"[fallback] no 3000-window; using nearest {hi-lo} reps around query {r}", flush=True)

    r, target, lo, hi = chosen
    rep_rows = rep[lo:hi][:N_REPS].astype(np.int64)
    shifts = (target - nmr[lo:hi][:N_REPS]).astype(np.float32)
    a, b = L.off[r], L.off[r + 1]
    qm, qp = L.cmz[a:b], L.cp[a:b]
    print(f"[query] rep_idx={r} n_rep={len(rep_rows)} q_peaks={len(qm)} tol={A.MZ_TOL} q_mass={target:.3f}", flush=True)
    # gauge how many reps actually carry peaks
    act = 0
    for j in range(len(rep_rows)):
        x, y = L.off[rep_rows[j]], L.off[rep_rows[j] + 1]
        if y > x:
            act += 1
    print(f"[reps] {act}/{len(rep_rows)} reps carry >=1 cleaned peak", flush=True)
    assert len(qm) >= MIN_Q_PEAKS, f"query has only {len(qm)} peaks"

    # warm both njit functions (JIT compile) on the real batch -- but on a copy
    # we keep warmups on full batch to compile the prange loop body properly
    sub = rep_rows  # full shape for compilation
    _ = A._search_shift_batch(qm, qp, sub, L.coff.astype(np.int64), L.cmz, L.cp, A.MZ_TOL, shifts)
    _ = F.search_shift_prange(qm, qp, sub, L.coff.astype(np.int64), L.cmz, L.cp, A.MZ_TOL, shifts)
    print("[warmup] both JIT'd on full batch", flush=True)

    coff = L.coff.astype(np.int64)
    cmz, cp = L.cmz, L.cp
    tol = A.MZ_TOL

    def bench(fn, n=3):
        best_t = 1e9
        best_out = None
        for _ in range(n):
            t = time.time()
            out = fn(qm, qp, rep_rows, coff, cmz, cp, tol, shifts)
            dt = time.time() - t
            if dt < best_t:
                best_t = dt; best_out = out
        assert best_out is not None
        return best_t, best_out

    print("[OLD] _search_shift_batch (serial) ...", flush=True)
    t_old, out_old = bench(A._search_shift_batch)
    print("[NEW] search_shift_prange (parallel) ...", flush=True)
    t_new, out_new = bench(F.search_shift_prange)

    same_shape = out_old.shape == out_new.shape and out_old.shape == (len(rep_rows),)
    idxs = [0, len(rep_rows) // 2, len(rep_rows) - 1]
    md_max = 0.0
    for i in idxs:
        md_max = max(md_max, abs(float(out_old[i]) - float(out_new[i])))
    full_diff = float(np.max(np.abs(out_old - out_new))) if same_shape else np.nan
    nz_new = int(np.count_nonzero(out_new))

    print("=" * 70)
    print(f"F-SPEED BENCHMARK v2 | {len(rep_rows)} reps | q_peaks={len(qm)}")
    print(f"  OLD serial   : {t_old*1000:8.1f} ms  ({t_old/len(rep_rows)*1000:.3f} ms/rep)")
    print(f"  NEW prange   : {t_new*1000:8.1f} ms  ({t_new/len(rep_rows)*1000:.3f} ms/rep)")
    print(f"  speedup      : {t_old/t_new:.1f}x")
    print(f"  shapes match : {same_shape} ({out_old.shape})")
    print(f"  spot-check (idx {idxs}) max |diff| = {md_max:.6f}")
    print(f"  full-array max |diff|         = {full_diff:.6f}")
    print(f"  nonzero outputs (new)         = {nz_new}/{len(rep_rows)}")
    print(f"  GATE (<=6s for 3000 reps)     : {'PASS' if t_new <= 6.0 else 'FAIL'}")
    print("=" * 70)

    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "FSPEED_REPORT.md"), "w") as f:
        f.write("# F-SPEED REPORT — measured (valid peak-rich query)\n")
        f.write(f"- reps: {len(rep_rows)}, q_peaks: {len(qm)}, tol: {tol}\n")
        f.write(f"- OLD `_search_shift_batch` (serial): {t_old:.3f} s ({t_old/len(rep_rows)*1000:.3f} ms/rep)\n")
        f.write(f"- NEW `search_shift_prange` (prange): {t_new:.3f} s ({t_new/len(rep_rows)*1000:.3f} ms/rep)\n")
        f.write(f"- speedup: {t_old/t_new:.1f}x\n")
        f.write(f"- shapes match: {same_shape}\n")
        f.write(f"- spot-check max |diff| (idx {idxs}): {md_max:.6f}\n")
        f.write(f"- full-array max |diff|: {full_diff:.6f}\n")
        f.write(f"- nonzero outputs: {nz_new}/{len(rep_rows)}\n")
        f.write(f"- gate (<=6s): {'PASS' if t_new <= 6.0 else 'FAIL'}\n")


if __name__ == "__main__":
    main()