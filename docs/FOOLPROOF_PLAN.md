# CASMI26 — FOOLPROOF REBUILD PLAN (v2, 2026-09-19, continuous-planning pass)
> Owned by Root. Every phase has a numeric gate on the SAME metric as the LB (MRR@25, InChIKey14
> canonical, Class-1/Class-2 simulation). Nothing advances without passing its gate on a holdout
> that does NOT lie. v2 change: Phase 1 is now FORK-FIRST (reproduce a proven 0.335, then diverge),
> not rebuild-from-scratch, and compute reality is budgeted explicitly.

## THE GROUND RULES (non-negotiable, from the 5-notebook teardown)
1. **MRR@25 is an exact-structure metric.** Retrieval is the game. Generation (Class-3) is only
   worth it AFTER retrieval rejoin — a generated SMILES that isn't the exact InChIKey14 tautomer
   scores 0.
2. **Pool quality >> ranking quality.** Big, dedup'd, well-windowed pool = biggest lever (±10ppm,
   InChIKey14-dedup, train∪COCONUT∪ChEBI/LIPID ~712k). NEVER cap aggressively (0.992→0.752 @80).
3. **Merge 4+ evidence channels, never one signal**: analog-propagation + gated library match +
   spectrum→fingerprint proba + in-silico fragmentation.
4. **A data-starved ranker is the #1 silent loss.** Train on SIMULATED Class-1/Class-2 rows
   (≥2,250 molecules × both classes).
5. **Fingerprint-model weights have SEEN the library** (0.76–0.82 seen vs 0.49 held-out). Two
   rankers (leak-safe + leaky), blended. Watch the FP channel's trust.
6. **Validate the way the LB scores**: Class-1/Class-2 simulation (remove structure from every
   library) + the 250 enveda-np held-out queries. Random splits OVERESTIMATE. Audit every number.
7. **Final 25 = metric-exact dedup** on tautomer-canonical InChIKey14 (RDKit 2026.3.3 grader). Only
   unique ranked candidates — NO padding (poison → 0.000 risk; rules allow <25, no penalty).
8. **OFFLINE-SUBMITTABLE at every checkpoint**: no pip/rdkit/internet in the kernel; ship pools/fps/
   peaks-lib as private datasets; pretest locally against a /kaggle/input mirror before pushing.
9. **Licensing greenlit**: COCONUT 2.0 + CFM-ID in-silico, LGPL runtime deps, ChemBERTa, MIT weights.
   Use them.
10. **Every gate number is real**: who measured it, on what split, with what code. Zero fabricated.

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

## PHASE 2 — Pool/retrieval quality (target: +0.02–0.08 on PHASE 1)
1. Measure **answer-in-top-25 rate** vs pool — how much of missing ~60% is pool-limited (Class-2)
   vs Class-3. This is the open community question; answering it deferentially steers Phases 2 vs 4.
2. Candidate-source sweep: COCONUT subsets, domain expansion, adduct-aware windows. Keep anything
   that raises top-25 retention WITHOUT wrecking analog-Tanimoto (PubChem = caution, cat 0.350).
3. **Adduct-shifted library similarity** (megayak): same mol as [M+Na]+ vs [M+H]+ keeps neutral
   losses. Adduct labels curated-clean — DON'T re-hypothesize, DO shift-match.
- Gate: Class-2 MRR up AND LB up. Stop when marginal < noise.

## PHASE 3 — Ranker/calibration polish (target: +0.02 on PHASE 2)
1. Feature audit: which of the ~31 features carry signal on Class-2 (library-miss)?
2. Multi-seed bag, dual-prior W, per-class Platt calibration BEFORE final blend.
3. Candidate-cap re-test at multiple windows (400 vs none) — confirm retention holds.
- Gate: discard anything that doesn't beat incumbent by ≥2× noise.

## PHASE 4 — Class-3 generation (target: first positive net; the differentiated lever)
**~39% of test, NO public winner — the open, winnable angle.**
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

## STATUS TRACKER (v2)
- [ ] PHASE 0 — kern v9 RUNNING, score PENDING (confirms clean submission)
- [ ] PHASE 1 — FORK prvsiyan → reproduce ≥0.30 → 3 surgical grafts (the big jump)
- [ ] PHASE 2 — pool/retrieval quality
- [ ] PHASE 3 — ranker/calibration polish
- [ ] PHASE 4 — Class-3 de-novo generation (differentiator, GPU budgeted)
- [ ] PHASE 5 — ensemble & final submit