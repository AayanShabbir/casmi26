# HANDOFF — CASMI26 session 2026-09-18 (start here next session)

Read this first in a fresh session. Everything below is verified, committed state —
do not re-derive.

## Isolated worktree & data
- Repo: `~/code/casmi26` (git, isolated). Data gitignored: `train.parquet`(2.8GB),
  `test.parquet`, `sample_submission.csv`, `data/` pool, `.venv`.
- venv: `~/code/casmi26/.venv` (rdkit 2025.09.2, numba 0.60, numpy, pandas,
  pyarrow, scipy, scikit-learn). **numba installed.**
- Kaggle data local, competition JOINED (user `aayanshabbir`), CLI
  `~/.local/bin/kaggle` 2.2.4 works (ACCESS_TOKEN auth).

## Goal
Top-10% medal (MRR~0.30+) on Enveda CASMI26 (400 mol, MRR@25, deadline
2026-12-14). Winner ~0.339. We reproduce + improve the public V17 winner pipeline.

## What is DONE + measured (all committed)
- **Winner dissected** (4-channel retrieval + 31-feat GBDT). Ground truth: 82.6%
  of winner's guesses are in train library; the real game is RANKING ~132
  mass-matched candidates, not novel-discovery. Plan: `PLAN.md`.
- **F-ANALOG (backbone) COMMITTED + MEASURED**: `src/analog_run_v2.py`,
  commit `8cbc4be`. **MRR@25 = 0.2125 | hit@25 = 62.75%** on valid 51-molecule
  holdout (train-structures pool). Recall proven; ranking weak. This is the
  foundation number.
- **Pool cached**: `data/trainpool.npz` (275,810 structs, consistent
  Morgan2+3+RDKit2048 1280-byte fp scheme, NOT tied to COCONUT fp_bits).
- **Key lessons learned (do not re-live):**
  - Fleet profiles (`hermes --profile X chat -Q -q`) WEDGE ~5/5 on heavy local
    runs — spawn MCP suite then idle at 0% CPU, never connect model. FLEET
    CANNOT execute CASMI heavy compute. Work happens IN-SESSION (root).
  - Dispatch law: brief = lean pointer to PLAN.md/doc, never inline payload;
    never `-Q -q` for execution (hides wedges).
  - Fleet lane: `agy-bridge` (gemini-3.8 at :8790) is the free lane; NOT `agy`
    (hangs). Engineer+checker profiles routed to agy-bridge (engineer config
    edited to AGY this session).

## NEXT (the actual work — highest priority)
1. **F-SPEED** — `src/f_speed.py` WRITTEN but NOT yet benchmarked (the user said
   stop this session). It's a prange-parallelized mass-shift similarity
   (`search_shift_prange`), replacing the serial `_search_shift_batch`
   (40ms/rep, >120s/3000). GATE: <=6s for 3000 reps. Callers currently import
   `analog_accel._search_shift_batch` — swap to `f_speed.search_shift_prange`
   (same contract). Benchmark in next session.
2. **F-RANKER** — `src/f_ranker.py` mostly written (feature extraction + HistGB
   on 120 train queries, same 60-mol holdout). Was blocked on sim speed.
   After F-SPEED passes, run it. Gate: beat 0.2125.
3. Then F-CONS / F-MASS / F-ALEIGN / F-EAERN (spec2vec) — each gated > incumbent.

## Savant
- Workspace `CASMI26 Molecule ID` (id 2277616941600993050) exists. Task
  `tid-96cf8ee` = F-ANALOG (re-scoped). KG node staged for plan.
- Fleet routing: checker + engineer on agy-bridge. AGY bridge healthy at :8790.

## Anti-patterns to avoid (learned the hard way)
- Do NOT waste time on stale result files (`docs/fanalog_measured` had stale
 0.000) — check mtime before trusting.
- COCONUT pool ∩ train = ZERO (natural vs synthetic) — a COCONUT-only candidate
  pool makes hold-out MRR trivially 0. Must use train-structure pool.
- `-Q -q` quiet dispatch HIDES wedges. For anything that needs recovery, run
  steerable / in-session.
- f_p schme must be SELF-CONSISTENT (candidate vs analog), need not match author's
  exact fp_bits (that is author-private and a rabbit hole).