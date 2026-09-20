# CASMI26 — FOOLPROOF REBUILD PLAN (v3, 2026-09-19, continuous-planning pass 2)
> v3: corrected class-share math (f₃≈0.55 majority, NOT 39%), inverted the lever ordering
> (ranking > recall — pool already saturated), and added the flagship's hard-won measured
> dead ends so we don't burn submissions/time re-discovering them. Data source: prvsiyan
> flagship's own measured teardown (read in full).
> Owned by Root. Every phase has a numeric gate on the SAME metric as the LB (MRR@25, InChIKey14
> canonical, Class-1/Class-2 simulation). Nothing advances without passing its gate on a holdout
> that does NOT lie. v2 change: Phase 1 is now FORK-FIRST (reproduce a proven 0.335, then diverge),
> not rebuild-from-scratch, and compute reality is budgeted explicitly.

## GROUND RULES (v3) — non-negotiable, from 5-notebook teardown + prvsiyan's measured numbers
1. **MRR@25 is an exact-structure metric.** Retrieval is the game. A generated SMILES that isn't
   the exact InChIKey14 tautomer scores 0.
2. **RANKING > RECALL (v3 correction).** The pool is ALREADY saturated — ρ≈1, COCONUT covers 99.6%
   of enveda-np-examples. The ceiling with any reasonable pool is ~0.43; every point to 0.43 is
   ORDERING, not coverage. **Do NOT spend time enlarging databases.**
3. **Merge 4+ evidence channels, never one signal**: analog-propagation + gated library match +
   spectrum→fingerprint proba + in-silico fragmentation. Each is an independent view.
4. **A data-starved ranker is the #1 silent loss.** Train on SIMULATED Class-1/Class-2 rows
   (≥2,250 molecules × both classes). Hold out by QUERY, never by row (candidates of one molecule
   must be on one side of the split).
5. **Fingerprint-model weights have SEEN the library** (0.76–0.82 seen vs 0.49 truly held-out).
   Two rankers (leak-safe + leaky), blended.
6. **Local validation is a FILTER, not a DECISION.** It catches broken things (retention, leaks,
   recall) but CANNOT rank two working configs. Config choices cost submissions. Audit every gate.
7. **Final 25 = metric-exact dedup** on tautomer-canonical InChIKey14 (RDKit 2026.3.3 grader).
   Unique ranked candidates only — NO padding (poison; rules allow <25).
8. **OFFLINE-SUBMITTABLE at every checkpoint**: no pip/rdkit/internet in the kernel; pools/fps/
   peaks-lib as private datasets; pretest locally against a /kaggle/input mirror before pushing.
9. **Licensing greenlit**: COCONUT 2.0 + CFM-ID in-silico, LGPL runtime, ChemBERTa, MIT weights.
10. **Every gate number is real**: who measured it, on what split, with what code. Zero fabricated.

## CLASS-SHARE MATH (v3 correction — from prvsiyan's LB algebra + measured pool coverage)
- **f₁ (Class 1) ≈ 0.162** — saturated; the 0.151 library-only submission already IS all of it.
- **f₂ (Class 2) ≥ 0.27** (hard bound from B = f₂·ρ = 0.271, ρ≤1); working estimate 0.27–0.30.
- **f₃ (Class 3) ≈ 0.55 — the MAJORITY.** Unreachable by ANY retrieval. This is the real prize and
  the field is at its raw start. (My v2 called it 39% — wrong; the leaderboard forbids more than
  ~0.43 from retrieval alone, and leaders cluster just above 0.35 as everyone converges on the same
  reachable fraction.)
- **Where we are:** 0.077 clean = we're not even extracting class-1/library value yet. Getting
  retrieval RIGHT is worth up to ~0.43. Generation (f₃) is the difference between ~0.43 ceiling
  and a real win.

## COMPUTE REALITY (budget — constrains everything)
- **Our Mac (16GB unified): NO usable GPU for this.** Pool/fingerprint/dataset *build* = CPU, fine
  (one-time). Heavy model train/predict (FPNet, transformer, large ranker) runs **ON KAGGLE GPU**.
- **Kaggle GPU quota = 30 H/week** (stated in discussion 742082). CPU runs are ~free but 9h cap.
  So: budget GPU hours — a flagship reproduction is ~1h GPU per run. Don't blow 30h on dead ends.
- Strategy: **local = cheap dataset build + validation harness + offline pretest. Kaggle = the
  actual expensive run only when locally green and gated.**
- Kill scope that can't run: do NOT plan a big local deep model. That lives on Kaggle GPU.

## PHASE 0 — Baseline rejoin (target: ≥0.077 clean)
**Status: `kern v9` (fixed writer) RUNNING on Kaggle → score pending. Current = 0.077.**
- Deliverable: clean, correct, offline submission; confirms the metric pipeline is sane.
- Gate: LB ≥0.077 → confirms plumbing; book the number.
- WHY: isolates "is our submission pipeline correct?" from "is our engine good?".
  A *clean* 0.077 tells us the engine is the problem, not the plumbing.
- **v2 note:** if v9 scores notably WORSE, the writer is broken — stop everything, fix it.

## PHASE 1 — FORK-FIRST rejoin (target: reproduce ≥0.30 on LB)
**THE structural fix vs v1: we do NOT rebuild the 4-channel engine ourselves.**
1. **Local fork of prvsiyan's flagship** (`analog-propagation-casmi-2026-baseline`, Apache 2.0).
   Pull its source (already in /tmp/nbs). Map its inputs:
   - `CASMI26 fp models late merged` (dataset: fp models single+merged)
   - `CASMI26 ranker training features (public)`
   - `ChEBI + LIPID MAPS candidates for CASMI26`
   - `COCONUT 2.0 candidates + fingerprints (CASMI26)`
   → these are PUBLIC datasets; mount all as `dataset_sources` (no rebuild). Offline-safe.
2. **Reproduce his 0.335 on the SAME public-LB run** (aim ≥0.30, accept ≥0.28 as first fork).
   - He ran CPU-only 1h7m; we run CPU first (free), GPU only if needed.
   - Local pretest: extract code cells → run against local /kaggle/input mirror → confirm 400-row
     clean output BEFORE pushing (same hard-won recipe).
3. **Graft the 3 proven teardown fixes that are drop-in**: metric-exact final-25 dedup (needs RDKit
   canonical — ship offline RDKit wheel like beraterolelk does), FPNet full-128-peaks (denpugovkin),
   leak-safe+leaky dual ranker (megayak). Each gated: only keep if LB up by ≥2×seed-noise (±0.012).
- Gate: **LB ≥0.30 → Phase 1 done.** (0.28 min acceptable first push.)
- This turns a multi-week rebuild into a fork → reproduce → 3 surgical grafts.

## PHASE 2 — RANKING quality (v3: rename + re-scope; target: +0.02–0.06 on PHASE 1)
**v3 correction: this was "Pool/retrieval quality" — recall is NOT the bottleneck (ρ≈1).**
1. **Second input VIEW, not a second model.** The largest isolated effect in the whole comp
   was **+0.019** from pairing a per-spectrum model with a merged-input model. Add a genuinely
   *different* reading of the spectra (different peak preprocessing / collision-energy-conditioned
   encoder / MS1-aware) — cheap vs training a bigger net, high value. (prvsiyan #1.)
2. **Better per-spectrum model.** The pair is only as good as its weaker half; ours is ~20k steps.
   Retrain the fp model longer w/ best-by-validation checkpoint + augmentation. (prvsiyan #2.)
3. **More ranker seeds** — free variance reduction vs 0.0072 noise floor. Tune folds held out by
   QUERY. (A second merged-input model of the same kind is redundant — measured −0.006.)
4. **Measure answer-in-top-25 rate** as a function of pool once (the open question), but do NOT
   chase it with database expansions — that is measured-dead (see dead ends below).
- Gate: LB up by ≥2×noise (≥0.012), twice, on two DIFFERENT submissions. Not local MRR.

## PHASE 3 — Ranker/calibration polish (target: +0.01–0.02 on PHASE 2)
1. Feature audit: which of the ~31 features carry signal on Class-2 (library-miss)? (Mass-error
   percentile and view-agreement were measured washes — don't chase.)
2. Multi-seed bag, dual-prior W, per-class Platt calibration BEFORE final blend. Re-sweep the
   class prior whenever a channel improves (the W optimum moved 0.50→0.45 as the model improved).
3. Candidate-cap re-test (400 vs none) at the ranker, NOT the window — window ±10 ppm is load-
   bearing, post-window caps cost Class-2 recall. Verify retention holds.
- Gate: config choice is settled by SUBMISSION (local validation cannot rank configs), and only
  kept if LB beats incumbent by ≥2×noise. Two identical-config runs first to soak the noise floor.

## PHASE 4 — Class-3 generation (target: first positive net; THE differentiated lever)
**v3 correction: f₃ ≈ 0.55 — the MAJORITY of the test. No public winner. This is the real prize.**
1. spectrum→SMILES de-novo generator (encoder-decoder transformer; ChemBERTa or MolMule-style
   decoder) fine-tuned on 2.5M train spectra. Output top-K candidate SMILES.
2. **CRITICAL GATING**: a generated candidate must clear an evidence threshold (FP proba + analog
   support + fragment score) to enter the list, replacing only the WEAKEST retrieval candidates —
   never a takeover. Metric-exact dedup against the retrieval list (poison prevention).
3. Honest validation: synthetic Class-3 holdout = molecules REMOVED from every library AND excluded
   from pool, spectra via CFM-ID in-silico. Require exact-InChIKey14 recovery at a measurable rate.
4. **Compute note (v2):** this is the GPU-heavy phase — spend Kaggle GPU budget here ONLY after
   Phases 1–3 bank their gains. Budget explicitly (~10–15 h/wk max, parallel to retrieval work).
- Gate: Class-3-only MRR >0 AND blended LB not below Phase 3. Grow from there.
- HONEST EXPECTATION: hard; exact-match generation is the field's open problem. Don't let it
  cannibalize retrieval.

## PHASE 5 — Ensemble & final submit (target: ≥0.34 clean)
1. Merge: retrieval pipeline + generator candidates (gated), multi-seed, calibrated.
2. Final-25 metric-exact dedup; verify 400 rows × valid-resolvable SMILES; pretest local
   full-pipeline; push; submit.
3. Only the best LB-scoring config ships. Document winner + runner-up.

---
## VERIFICATION DISCIPLINE (every phase)
- Each claimed number: run the exact evaluation on a leakage-audited split. A number is a LIE until
  it survives an adversarial re-check.
- Push to Kaggle ONLY after the same code ran green locally against a /kaggle/input mirror.
- LB deltas < ±0.006 seed noise; phase "done" only when its gate is beat by ≥2× noise.
- Every push records: config hash, local gate number, expected LB range. Compare.

## FLOOR CONTRACT (Phase A — LOCKED, 2026-09-20)
- **Banked floor = 0.328** (ref 56371855 fork / 56375106 auto). Artifact: `submissions/floor-0.328.csv` (400 rows, 0 nulls, verified).
- **Submit rule: NOTHING below 0.328 ships.** The floor is the default fallback for any future submission:
  if a run can't be shown to beat it by ≥2×noise (±0.012) with a plausible mechanism, the floor is what goes in.
- Every new experiment's baseline gate is this floor, not a weaker intermediate.

## STATUS TRACKER (v4)
- [x] PHASE 0 — kern v10 COMPLETE: **LB 0.077**, clean format, plumbing CONFIRMED (Session 2026-09-19)
- [x] PHASE 1 — FORK of prvsiyan reproduced: **LB 0.328** (his best 0.335; within seed noise). GATE ≥0.30 SMASHED.
- [x] PHASE 1b attempt #1 (metric-exact dedup) — **NO-OP**: output byte-identical to fork. Pool already
  InChIKey14-deduped at build; raw top-25 never had canonical dupes (0/9609). NOT submitted. Learned:
  dedup graft was already baked into the pool build.
- [ ] PHASE 2 — RANKING quality (second input view, better per-spectrum model, seeds)...[truncated]
- [ ] PHASE 3 — ranker/calibration polish (config by submission)
- [x] PHASE 4 feasibility PROBE (2026-09-19) — **NO-GO**: spectrum→SMILES seq2seq, leak-free
  structure-holdout, 4000 steps, loss 4.6→0.62 but **val exact 0/120, valid 0/120** at every step.
  Naive auto-regressive generation of held-out structures does not learn in this framing. **Parked
  Phase 4** pending a fundamentally different framing (fg-conditional diffusion / strong pretrained
  ChemBERTa-aligned decoder) that shows signal in its own probe first. Full result → `docs/GEN-PROBE-RESULT.md`.
- [ ] TRACK A (now): ranking/ordering polish on 0.328 engine (better per-spectrum fp model, second
  input view, seeds) → target 0.34–0.35 (realistic pool ceiling ~0.43)
- [x] TRACK A exp1 (fp v6 ensemble swap, 2026-09-20) — **LB 0.320, no gain** vs 0.328 baseline
  (−0.008 ≈ 1.3×seed noise, slightly negative). Bigger/newer fp weights do NOT help ordering on the
  current engine. **Lever dead — do not re-test weights.** Remaining Track-A option = genuinely new
  input VIEW (collision-energy-conditioned / MS1-aware), not more of the same model family.
- [ ] PHASE 5 — ensemble & final submit

---
## ⛔ MEASURED DEAD ENDS (from prvsiyan's teardown — do NOT burn submissions re-discovering)
| Idea | Measured result |
|---|---|
| Add all PubChem isomers to pool | 0.52→0.35 Class-2 (catastrophic) |
| Admitting model's top-50 PubChem | 0.337 — still bad |
| Soft consensus-fp over analogs | 0.43 vs 0.52 max-over-analogs |
| Per-instrument analog-sim normalisation | 0.517 vs 0.521 |
| Confidence gate 'answer-not-in-pool' | AUC 0.627 — too weak to route on |
| 2nd model of SAME input view | 0.335→0.329 (redundant) |
| Test-time avg over per-spectrum+merged | merged-only 0.516 > mean 0.509 |
| Re-weight analogs by predicted-fp fit | 0.521→0.512 |
| 3 ref spectra/structure instead of 1 | 0.52→0.49 |
| NP-likeness prior | 0.037 (worse than random) |
| Explicit molecular-formula pred | only 1.4× candidate reduction |
| Library sim as fixed additive term | Class-2 0.52→0.27 |
| Coarse cap 80 by lib_sim×100−\|Δmass\| | **−0.055 LB** (kills 24% Class-2 recall) |
| z-score blend instead of GBM | 0.606 vs 0.620 |
| Mass-error percentile feature | wash (helps only Class-1) |
| Both model views + agreement features | 0.3042→0.3024 wash |
| Targeted derivative enumeration (±O,±CH₂…) | 0.337→0.335 (starkhushi) |
| more analogs >80 | flat (80→3709→100→3705) |

**Still-open levers (spend here):** second input VIEW (+0.019 max effect), better per-spectrum
model, more seeds, Class-1 recall for cross-instrument cases, and (the real prize) Class-3
generation. Ranking, not recall; the ceiling with this pool is ≈0.43.