# CASMI26 — Living Plan (continuous-planning, fleet-structured)

**Goal:** Top-10% medal (target MRR ~0.30+) on Kaggle Enveda CASMI26
(400 molecules, MRR@25) by reproducing the public winner pipeline (0.335-0.339)
then beating it by grafting stronger open-source channels. Deadline 2026-12-14.

## MEASURED grounds (corrected this pass — earlier numbers were wrong)
- **Winner output analyzed:** 400×25, 100% valid, mass-correct (+1.008 Da =
  [M+H]+), 82.6% of guesses in train library, 17.4% novel.
- **RETRACTION — the 0.108/0.0075 Ch1 numbers are INVALID.** Fleet F-BASE
  (library search, twins-excluded) ruled 0.0075 locally — that OVER-tests the
  17%-novel case. Winner diagnostics (Quad-Channel, public 0.339) show
  **Ch1 library sim = 1.0 for 100% of real test molecules**. Retrieval is ~solved
  by the matched library. The real game: among **~132 mass-window candidates**,
  rank the CORRECT one first — a RANKER problem, not a library problem.
- **Analog propagation is the real backbone (winner-measured):**
  - single-channel Class-2 MRR **~0.52** (fair held-out, natural products AND
    569-obscure structures → 0.516, generalises).
  - prvsiyan analog-baseline public-LB ladder: spectral-only 0.151 →
    +analog+ranker 0.233 → +class-weight 0.245 → +fragmentation 0.266 →
    +fingerprint 0.299 → +merged-model/seed-rank **0.335**.
- **Gate law:** every channel upgrade MUST beat the incumbent measured MRR to
  ship (head-to-head, ConnectX discipline). No swap ships on a gut feeling.
- **KEY LEAK (Two-Rankers, interview-grade):** public FPNet weights were trained
  on most library structures → score 0.76-0.82 on seen sets, only 0.49 on truly
  held-out. A ranker trained on them over-trusts the fingerprint channel.
  MITIGATION (winning): two rankers (leak-free + leaky), blend ~0.65/0.35.
- **MEASURED DEAD-END:** DreaMS embeddings as analog channel only ties Class-2
  (0.58 vs 0.57-0.63) and helps only Class-1. Do NOT graft DreaMS.

## Attack findings THIS pass
1. **Killed a fake floor.** F-BASE's ≥0.108 gate was measuring the wrong
   distribution (novel-only). The real, winner-proven backbone is analog (0.52
   single-channel). Re-scoped: F-BASE gate = reproduce analog ≥~0.50 on a fair
   Class-2 held-out. This is what actually buys MRR.
2. **Killed dead scope earlier than before:** the whole Ch1-retrieval-rung chain
   (F-CONS/F-MASS/F-ALIGN) is *lower value* than analog — the winners ladder
   shows analog+ranker carries 0.233 of the 0.335. Reordered: ANALOG first.
3. **Identified the leak trap** that would silently poison our ranker — fixed by
   the two-ranker blend (their proven mitigation), not by pretending no leak.
4. **No single 3-mo estimate is checkable** → every phase has an EXIT GATE number.

## Fleet-structured phases (each = dispatched persona, handoff-gated)
Each phase: `researcher` (find/measure) → `architect` (spec) → `engineer`
(build) → `checker` (judge vs incumbent). Gate must PASS to hand off. Savant
workspace `CASMI26` (2277616941600993050) + KG track all.

**F-ANALOG (fleet: engineer + checker) — THE BACKBONE, build first**
- Reproduce mass-shifted analog propagation: ±200 Da window, `sim(a)^p ·
  Tanimoto(fp_c, fp_a)` with p=4, on the COCONUT∪train pool (711,705).
- GATE: **reproduce single-channel MRR ~>= 0.50** on a fair Class-2 held-out
  (enveda-np-examples style: remove a natural product's spectra, match to the
  rest). Proven achievable (winner: 0.521/0.516). Source:
  `prvsiyan/analog-propagation-casmi-2026-baseline` (73 votes, has the kernel).

**F-RANKER (fleet: architect→engineer→checker) — the 0.233 achievement**
- Build the 31-feature GBDT ranker (2 class-priors × 4 seeds) from
  `rank_train.npz`-equivalent; train rows from a 7-query-set simulation
  (2,250 mols, each twice: Class-1 + Class-2) — fixes the data-starved flaw.
- GATE: predicted-LB on enveda-np-examples ~0.233 (the proven +analog+ranker
  step). Source: `megayak/casmi26-two-rankers-one-engine`.

**F-CLASS (fleet: checker) — class-weight calibration**
- Leaderboard-calibrated W1 priors (0.30, 0.60). GATE: ~0.245 (their +calib).

**F-FRAG (fleet: engineer→checker) — in-silico fragmentation (MetFrag-lite)**
- Adduct-aware bond-break scoring. GATE: ~0.266 (proven step).

**F-FP (fleet: engineer→checker) — fingerprint model channel**
- **LEAK-SAFE**: use prvsiyan's public FPNet weights BUT train our ranker on
  leak-free rows, or run two rankers + blend 0.65/0.35 (their proven mitigation).
  GATE: ~0.299 single-ranker / up to 0.335 with merged+seed blend.

**F-SHIP (fleet: engineer + checker) — package + submit**
- Isolated worktree `~/code/casmi26` (already git-seeded), venv+Dockerfile,
  notebook with pinned deps + bundled models. GATE: notebook output == local
  script output; private-LB pulls.

**DELAYED (explicitly out of critical path, GPU-gated):** SIRIUS/CSI:FingerID
(Ch3) and MIST/trained-FPNet (Ch4) — only if F-ANALOG→F-FP leaves headroom and a
GPU lane exists. Not promised, not needed for medal.

## Predicted final outcomes (honest ranges)
- F-ANALOG→F-FP realistic cumulative: **~0.30-0.335 MRR → top-10% medal, likely
  8th-15th of ~670 teams** — a genuine, defensible mass-spec-ML resume line.
- With SIRIUS/MIST headroom: up to ~0.34+ — cash-podium territory, not promised.
- Floor we betray if we stop after F-ANALOG: ~0.52-single × 33% ≈ 0.17-0.20
  (beats a from-scratch baseline but loses the medal).

## Open questions / risks
- Can we reproduce analog at ~0.50 without their exact pool ordering? The COCONUT
  pool + fp_bits are public (`prvsiyan/coconut-casmi26-candidates`); the analog
  math is in the public notebook. High confidence, must verify on-disk.
- Leak management is the #1 correctness risk — two-ranker blend or leak-free
  rows is the acceptance gate, not optional.
- LB public/private noise ±0.02 → judge locally vs a fair Class-2 holdout.
- No local GPU → everything above is CPU-only (winner ran their full engine in
  ~47 min CPU). Only SIRIUS/MIST need GPU; they're delayed.

## Priority next improvement (queued)
Execute **F-ANALOG** (the real backbone) — reproduce mass-shifted analog
propagation to single-channel ~0.50 on a fair Class-2 held-out. That number
validates the whole ladder before we build the ranker. Everything else waits on it.