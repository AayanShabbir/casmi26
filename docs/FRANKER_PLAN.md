# F-RANKER — combined-evidence GBDT candidate ranker (CASMI26)

**What it is:** the winner's `HistGradientBoosting` ranker replaces the hand-tuned
`max(sim^4·Tanimoto)` single-score ordering. It takes ~31 candidate features,
*trains* on simulated positive/negative rows, and ranks the mass-window candidates.

**Why it matters (measured gap):** F-ANALOG = MRR 0.212 (analog single-channel only).
The winner's ladder shows the ranker + extra channels carrying 0.233→0.335. F-RANKER
is the first real step on that path: learn the ordering instead of hardcoding it.

**Current state (committed):**
- `analog_run_v2.py` builds the train-only pool (cached 275k fps) + computes per-candidate
  `lv` (library sim), `analog sim values`, `sim^4·Tanimoto`. MRR 0.2125 / hit@25 62.75%.
- The raw analogs + Tanimoto per candidate are already computed — F-RANKER consumes them.

**Phase plan (in-session, incremental):**
1. **Feature extraction** — per candidate build a feature vector (library-sim, analog-sim
   value + rank, tanimoto max/mean, sim^p variants, log(candidate count), mass error).
   Cheap; reuses the v2 loop's already-computed `sims` + `tan`.
2. **Training rows** — from held-out train molecules: the true answer = POSITIVE row;
   the other mass-window candidates = NEGATIVE rows (same as the winner's simulation).
   Write `data/rank_train.npz`.
3. **Train HistGB** (sklearn, small, CPU-fast): fit on (features, is-truth), 2-prior×
   4-seed bagging per winner.
4. **Rank + measure** — apply to held-out, MRR@25. GATE: **> 0.2125** (must beat
   the incumbent analog-only ordering) or it doesn't ship.

**Acceptance:** `python src/f_ranker.py` prints `[F-RANKER] MRR@25 = ...` and a PASS/FAIL
vs 0.2125. Real numbers only.