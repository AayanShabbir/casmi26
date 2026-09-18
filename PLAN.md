# CASMI26 — Living Plan (continuous-planning, fleet-structured)

**Goal:** Top-10% medal (target MRR ~0.30+) on Kaggle Enveda CASMI26
(400 molecules, MRR@25) by reproducing the public V17 winner (~0.30-0.336)
then beating it by grafting stronger open-source channels on. Deadline 2026-12-14.

## Measured grounds (verified, data-driven — never vibes)
- **Winner output analyzed:** 400×25, 100% valid, mass-correct (top-1 +1.008 Da =
  [M+H]+), 82.6% of guesses are in the train library, 17.4% novel.
- **Retrieval floor is MEASURED, not guessed:**
  - dumb 1-Da cosine any-mass: **MRR 0.027** (3.5% hit@25)
  - + 25-Da precursor window: **0.071** (10.8%)
  - + real Ch1 recipe (entropy-weighted, ±8.5 ppm, peak align): **0.108** (17.5%)
- **Gate law:** every channel upgrade MUST beat the incumbent measured MRR to
  ship. Never a swap that doesn't win head-to-head (ConnectX discipline).

## Attack findings THIS pass (the brutal part)
1. **The plan was not fleet-adequate — it was a linear TODO.** Fixed: restructured
   into persona-ownable, handoff-gated phases below (researcher→architect→
   engineer→checker per slice), so the fleet actually parallels, not serializes.
2. **False-credibility risk that dies in review:** "we reproduced the winner's
   Ch1" would be a lie — our 0.108 uses a 0.1-Da grid, far below the winner's
   Ch1-MRR 1.000. Added a HONEST-CEILING marker + the fine-alignment step as a
   first-class gate so we never oversell.
3. **Dead scope killed:** de-novo SMILES generation (MSNovelist) is GPU-bound,
   uncertain, low-signal per the 82.6%-retrievable data. Demoted from the plan;
   retrieval+analog are the load-bearing 82%. State it flat, don't pretend.
4. **Cheap/good reordered first:** consensus fusion (Rung 1, code-only) is the
   highest-ROI-per-hour step and it comes first. Expensive SIRIUS/MIST is gated
   behind measured retrieval headroom so we never burn days on a dead channel.
5. **No single 3-mo estimate is checkable** → every phase has an EXIT GATE number.

## Fleet-structured phases (each = dispatched persona, handoff-gated)
Each phase dispatches a real profile: `researcher` (find/measure) → `architect`
(spec) → `engineer` (build) → `checker` (judge vs incumbent). Gate must PASS to
hand off. Savant workspace `CASMI26` (2277616941600993050) + KG track all.

**F-BASE (fleet: engineer + checker) — assemble, one runner, today**
- Pull COCONUT pool (`prvsiyan/coconut-casmi26-candidates`) + fp_bits (RDKit
  2026.03.3), build `rank_train.npz` from train, one `run.py` that does
  Ch1+Ch2 end-to-end.
- GATE: clean-script Ch1 MRR reproducibly >= 0.108 (must reproduce the floor
  we measured, proving correctness before any upgrade).

**F-CONS (fleet: engineer + checker) — Rung 1: multi-spectrum consensus**
- Merge each molecule's 3 collision-energy spectra → 1 consensus query
  (winner `_merge_peaks`), re-measure.
- PREDICTED MRR: **0.12-0.16**. GATE: > 0.108 (any lift = keep; expect strong).

**F-MASS (fleet: engineer + checker) — Rung 2: adduct-aware + tighter mass**
- Correct neutral mass (precursor − proton), apply the +1.4 ppm timsTOF offset,
  ±8.5 ppm window on neutral; adduct-aware.
- PREDICTED MRR: **+0.01-0.03** on top of F-CONS. GATE: > incumbent.

**F-ALIGN (fleet: researcher→engineer→checker) — Rung 4: exact peak alignment**
- Replace 0.1-Da grid with exact m/z alignment @ 0.01 Da + true entropy
  weighting. Targets the honest Ch1 ceiling (winner = 1.000 on lib hits).
- PREDICTED MRR: **0.15-0.20** combined. GATE: > incumbent.

**F-LEARN (fleet: researcher→architect→engineer→checker) — Rung 3: learned sim**
- Graft **spec2vec** + **ms2deepscore**; embedded-space retrieval beats cosine.
  Researcher pins pretrained weights/bundling (internet-off constraint).
- PREDICTED MRR: **0.20-0.26**. GATE: > incumbent, clock < 9h, internet-free.

**F-ANALOG (fleet: engineer→checker) — Rung 5: analog propagation**
- ±200 Da mass-shift + scaffold Tanimoto (winner Ch2, Class-2 0.612). Catches
  the 17.4% novel. PREDICTED MRR: **0.26-0.30**. GATE: > incumbent.

**F-SHIP (fleet: engineer + checker) — P2: package ours**
- Isolated worktree `~/code/casmi26`, venv+Dockerfile, notebook pinned deps +
  bundled models, submit. GATE: submission equals local script output,
  private-LB pulls.

**DELAYED (explicitly out of critical path, GPU-gated):** SIRIUS/CSI:FingerID
(Ch3) and MIST/trained-FPNet (Ch4) — only if F-ANALOG leaves headroom and a GPU
lane exists (Kaggle notebook ≤9h). Not promised, not needed for medal.

## Predicted final outcomes (honest ranges)
- F-BASE→F-ANALOG realistic cumulative: **~0.26-0.30 MRR → top-10% medal, likely
  8th-15th of ~186-670 teams**, worth a genuine mass-spec ML resume line.
- With SIRIUS/MIST headroom: **up to ~0.32-0.33** — cash-podium territory but
  not guaranteed; de-novo tail demoted for a reason.
- Floor we betray if we stop at F-CONS: ~0.12-0.16 (still beats baseline, but
  not worth the winter).

## Open questions / risks
- Ranker feat-scheme must match `rank_train.npz` → F-BASE locks it before
  generating (P0 decision).
- LB public/private noise ±0.02 → judge locally vs train-holdout, report both.
- No local GPU → learned sim (spec2vec CPU-ok; ms2deepscore needs small GPU) may
  train/run on Kaggle notebook (≤9h) or rig; F-LEARN researcher confirms before
  F-LEARN engineer starts.

## Priority next improvement (queued)
Execute F-BASE then F-CONS (consensus fusion) — the cheapest, highest-ROI,
code-only upgrade, directly measured against the 0.108 floor. Everything else
waits on those two numbers.