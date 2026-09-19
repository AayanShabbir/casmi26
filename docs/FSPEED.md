# F-SPEED — make the analog similarity fast (blocks F-ANALOG + F-RANKER)

## The measured problem (do not re-derive)
`_search_shift_batch` (`src/analog_accel.py`) is the bottleneck. A batch over
3,000 reps took **>120 s** (~40 ms/rep). F-RANKER calls it for ~180 queries
(120 train + 60 eval) each over a wide +-200 Da rep window -> a 16-min+ grind.
This blocks every later phase. It is a compute wall, not a correctness one.

## Why it is slow — two root causes (verified)
1. **No real parallelization.** The function is `@njit(parallel=True)` but its loop
   body never parallelizes (NumbaPerformanceWarning "no transformation for parallel
   execution was possible"): it's a plain serial `for k in range(len(rep_rows))` with
   no `prange`. So it runs single-threaded and gained nothing from Numba's parallel
   target.
2. **No candidate pre-pruning.** It runs full `_entropy_sim` on every rep row in the
   window. The winner's `search_shift` skips low-value reps before the entropy math
   (keeps top-N peaks, prunes candidates with no aligned peaks).

## The fix (target: >20x, winner-class ~47-min full engine on CPU)
Rewrite `_search_shift_batch` so that:
- The loop over `rep_rows` is `with prange(...)` for real multicore speed.
- **Pre-prune** before entropy: for each rep, cheaply check peak-overlap against the
  query (e.g. a coarse 1-Da-binned intersection count on the cleaning already done);
  skip reps whose overlap is below a floor (e.g. <2 shared peaks) before the
  `_entropy_sim`. Keep exact same return contract (float32 array len==len(rep_rows)).
- Add an early-exit in `_entropy_sim` when merged peak count crosses a bound (the
  merged buffer is the cost).
- Keep `fastmath=True`, `cache=True`.

Write it to `src/analog_accel.py` (replace the function, keep everything else).
Verify by timing the SAME 3,000-rep benchmark below — target **<= 6 s** (>=20x):
   `time python -c "import sys;sys.path.insert(0,'src');import analog_accel as A,time,numpy as np;_=A.Lib('train.parquet');r=10;a,b=_.off[r],_[...]..."`
Use the exact benchmark from the diagnosis above (3000 reps).

## Acceptance / gate
Run the 3,000-rep timing. Report the new wall time. PASS iff **<= 6 s** and
`_search_shift_batch` still returns identical-shaped output (spot-check 3 reps match
the pre-change values to within 1e-3). Write `docs/FSPEED_REPORT.md` with the old/new
timing + the pre-prune threshold you chose. Real numbers only.