# CASMI26 — FOOLPROOF REBUILD PLAN (v1, 2026-09-19)
> Owned by Root. Every phase has a numeric gate on the SAME metric as the LB (MRR@25,
> InChIKey14 canonical, Class-1/Class-2 simulation). Nothing advances without passing
> its gate on a holdout that does NOT lie. Ordered by marginal score first (retrieval
> rejoin >> generation).

## THE GROUND RULES (non-negotiable, derived from the 5-notebook teardown)
1. **MRR@25 is an exact-structure metric.** Retrieval is the game. Generation (Class-3) is
   only worth it AFTER retrieval rejoin, because it's exact-match — a generated SMILES that
   isn't the exact InChIKey14 tautomer is 0 points.
2. **Pool quality >> ranking quality.** A big, dedup'd, well-windowed candidate pool is the
   single biggest lever (±10ppm, InChIKey14-dedup, train∪COCONUT∪ChEBI/LIPID ~712k).
   NEVER cap candidates aggressively (caps silently kill Class-2 recall: 0.992→0.752 @80).
3. **Merge 4+ evidence channels, never one signal**: analog-propagation + gated library match
   + spectrum→fingerprint proba + in-silico fragmentation. Each is an independent view.
4. **A data-starved ranker is the #1 silent loss.** Train on SIMULATED Class-1/Class-2 rows
   (2,250 molecules × both classes), never on a handful.
5. **Fingerprint-model weights have seen the library** (leak 0.76–0.82 on seen vs 0.49 held
   out). Run TWO rankers (leak-safe + leaky) and blend. Watch the FP channel's trust.
6. **Validate the way the LB scores**: Class-1/Class-2 simulation (remove structure from every
   library) + the 250 enveda-np held-out queries. Random splits OVERESTIMATE. Check every
   claimed gate number for leakage before believing it.
7. **Final 25 = metric-exact dedup** on tautomer-canonical InChIKey14 (same canonicalisation as
   RDKit 2026.3.3 grader). Only unique, ranked candidates — NO padding to 25 with junk/dupes.
8. **Fully offline submittable** at every checkpoint: no pip/rdkit/internet in the kernel; ship
   precomputed pools/fps/peaks-lib as private datasets; pretest locally against a local mirror
   of /kaggle/input before pushing. Every push = previous local-green output verified.
9. **Licensing is greenlit**: COCONUT 2.0 + CFM-ID in-silico spectra, LGPL runtime deps,
   ChemBERTa, MIT pretrained weights all allowed. Use them.
10. **Every gate number is real**: state who measured it, on what split, with what code.
    Zero fabricated scores.

## PHASE 0 — Baseline rejoin (target: ≥0.20, ideally 0.28–0.33)
**Status: `kern v9` (fixed writer) submitted, score pending. Current = 0.077.**
- Deliverable: a clean, correct, offline submission that scores what our CURRENT engine is
  actually worth (all candidates unique, no poison). Confirms the metric pipeline is sane.
- Gate: LB score ≥ 0.077 → confirms fix helped; book the number. If it's WORSE, investigate
  the writer immediately.
- WHY: isolates "is our submission pipeline correct?" from "is our engine good?".
  A submission that scores 0.077 cleanly tells us the engine is the problem, not the plumbing.

## PHASE 1 — Adopt the 4-channel retrieval pipeline (target: 0.30–0.34)
**This REPLACES F-ANALOG/F-RANKER. Source: prvsiyan Apache-2.0 flagship (+4 teardown fixes).**
1. **Pool build**: train ∪ COCONUT 2.0 ∪ ChEBI ∪ LIPID MAPS, InChIKey14-dedup, ±10 ppm window.
   Precompute fingerprints (Morgan/RDKit locally), neutral masses, SMILES map. Ship as private
   datasets (~offline-safe). VERIFY: median ≥50 candidates/molecule, truth-in-pool == 100%
   on the Class-1/Class-2 holdout.
2. **4 channels** (each returns a per-candidate score):
   - [C1] mass-shifted **analog propagation** (Class-2 engine). sim(a)^p · Tanimoto(fp_c, fp_a).
   - [C2] **library match** — cosine/greedy @10ppm, gated (only fires when a library hit exists).
   - [C3] **spectrum→fingerprint transformer/FPNet** proba (predict chemistry from spectra).
   - [C4] **in-silico fragmentation / MetFrag-lite** (adduct-aware; Na/K/NH4/Cl/formate carriers).
   VERIFY each channel independently on separated holdout before merging.
3. **Ranker**: multi-seed HistGBR (depth 6, ~500 iter, LR 0.03), trained on simulated Class-1 +
   Class-2 rows (≥2,250 mols), W_A dual-prior averaged. **Two rankers** (leak-safe + leaky) blended.
4. **Final-25**: metric-exact tautomer InChIKey14 dedup, unique, ranked, no padding.
5. **Gates**: Class-1 MRR ≥0.85, Class-2 MRR ≥0.60 on simulation; LB ≥0.30.
   NOTE: our current pool lacks COCONUT — this is the big structural jump from 0.077.

## PHASE 2 — Pool/retrieval quality push (target: +0.02–0.08 on PHASE 1)
1. Measure **answer-in-top-25 rate** as a function of pool — find how much of the missing ~60%
   (of the visible test's ~0.40 ceiling) is pool-limited (Class-2) vs Class-3.
2. Candidate-source sweep: COCONUT subsets, domain expansions, adduct-aware windows. Keep
   anything that raises top-25 retention WITHOUT wrecking analog-Tanimoto (PubChem = caution).
3. **Adduct-shifted library similarity** (megayak): same mol as [M+Na]+ vs [M+H]+ keeps neutral
   losses. Adduct labels are curated-clean — DON'T re-hypothesize, but DO shift-match.
4. dedup'd, conditional (3) — VERIFY gates +0 on Class-2 specifically.
- Gate: Class-2 MRR up, LB up. Stop when marginal return < noise (±0.006 seed noise).

## PHASE 3 — Ranker / calibration polish (target: +0.02 on PHASE 2)
1. Feature audit: which of the 31 features actually carry signal on Class-2 (library-miss)?
2. Multi-seed bag, dual-prior W, per-class calibration (Platt) BEFORE the final blend.
3. Candidate-cap re-test at multiple windows (400 vs none) — confirm retention holds.
- Gate: anything that doesn't beat incumbent by ≥2×seed-noise (±0.012) is discarded.

## PHASE 4 — Class-3 generation (target: first positive net, then grow)
**THE differentiated, unwon lever ~39% of test, no public winner.**
1. Frame: spectrum→SMILES de-novo generator (encoder-decoder transformer, ChemBERTa or a
   MolMule-style decoder) fine-tuned on 2.5M train spectra. Output = its top-K candidate SMILES.
2. **CRITICAL GATING**: a generated candidate must clear an evidence threshold (FP proba + analog
   support + fragment score) to enter the list, replacing only the WEAKEST retrieval candidates —
   never a full takeover. Poison prevention: metric-exact dedup against retrieval list.
3. Validate: a synthetic Class-3 holdout = molecules REMOVED from every library AND excluded from
   pool, with spectra generated (CFM-ID in-silico). Require the generator to recover exact
   InChIKey14 in a measurable fraction — this is the honest Class-3 simulation.
4. Gate: Class-3-only MRR > 0 (any), and blended LB not below PHASE 3. Grow from there.
- HONEST EXPECTATION: this is hard; exact-match generation is the field's open problem. Budget
  accordingly. Do NOT let it repay the retrieval phases.

## PHASE 5 — Ensemble & final submit (target: ≥0.34 clean)
1. Merge: retrieval pipeline + generator candidates (gated), multi-seed, calibrated.
2. Final-25 metric-exact dedup; verify 400 rows × valid InChIKey14-resolvable SMILES; pretest
   locally full-pipeline; push; submit.
3. Only the best LB-scoring config ships. Document the winner + runner-up.

---
## VERIFICATION DISCIPLINE (applies to every phase)
- Each claimed number: run the exact evaluation script on a leakage-audited split. A number is
  a LIE until it survives an adversarial re-check (checker-style).
- Any push to Kaggle is only after the SAME code ran green locally against a /kaggle/input mirror.
- LB improvements below ±0.006 are noise; a phase is "done" only when its gate is beat by ≥2× noise.
- Every kernel push records: config hash, local gate number, expected LB range. Compare.

## STATUS TRACKER
- [ ] PHASE 0 — kern v9 submitted, score PENDING (should confirm clean submission)
- [ ] PHASE 1 — 4-channel rebuild (biggest jump, starts NOW after Phase 0 confirms)
- [ ] PHASE 2 — pool/retrieval quality
- [ ] PHASE 3 — ranker/calibration polish
- [ ] PHASE 4 — Class-3 generation (differentiator)
- [ ] PHASE 5 — ensemble & final submit