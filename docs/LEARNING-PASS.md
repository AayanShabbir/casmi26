# CASMI26 — LEARNING PASS (2026-09-20) + revised next steps
> Deep-read the whole competition after hitting stalemate on Track A. Sources: leaderboard,
> 6 discussion threads (plateau-bisection, six-variants, EDA-x10, adducts), and pulled source
> of haideptry "0.339 Top 1" + inversion's de-novo tutorial.

## WHERE THE FIELD IS (the real state)
- Everyone public is plateaued **0.330–0.337 = ONE noise band** (±0.006 seed). starkhushi ran 6
  variants: best single 0.337, worst 0.323 — all inside noise. The public pipeline has converged.
- **Top of LB rose to 0.396** (Ozymandias) and 0.376 (Randy), 0.363 (DDM) — *above* the ~0.40
  "perfect-rerank ceiling" Victor measured. So the leaders are either (a) extracting ~90%+ of the
  top-25 ceiling via near-perfect ordering, (b) a larger pool (higher ceiling), or (c) a working
  generation component. DancingLumberjack's CV math insists it's **RANKING**, not pool/generation.
- Victor's probe bound: reordering the existing top-25 is worth at most **+0.076**; we extract 81%
  of a ≈0.40 ceiling, leaders ~90%. **Pool-quality is the other lever** — but PubChem and derivative
  enumeration both measured-harm.
- Honest reality: **our 0.328 is the pack.** Getting to 0.35 is squeezing ordering; the 0.396 leader
  is something structurally different we haven't reverse-engineered.

## WHAT I LEARNED THAT CHANGES THE PLAN
1. **Validation is the #1 silent failure — and ours overestimated.** starkhushi: random structure
   split is "far too easy — near-identical molecules leak across it" (local top-1 0.694 then LB
   0.330). Never trust a local ranking number; only the LB orders configs.
2. **Domain gap is instrument+chemistry, not just one.** Test is 100% timsTOF; the public fp models
   score ~0.80 on trained libraries but ~0.49 on held-out timsTOF natural products. enveda-180 is a
   *chemical island* (instrument match ≠ chemistry match); enveda-np-examples is the small bridge.
3. **De-novo is NOT dead — my probe was too weak.** inversion ships a real **768-dim/12-head
   PyTorch-Lightning spectrum→SMILES transformer** (BPE-tokenized, 6-layer enc+dec, generates 25
   candidates/molecule, candidate-pooled across spectra, inchikey14-dedup, generation-score ranked).
   That's a fundamentally different (and far larger) framing than my 256-dim probe. Not proven on LB
   (tutorial), but it's the "different framing" our probe said to require before re-opening Phase 4.
4. **Multiple spectra per molecule = underused evidence.** Test gives 1–16 (median 3); positive AND
   negative spectra of the same molecule carry complementary fragments; multi-adduct neutral-mass
   consensus is a rock-solid anchor (1.3 ppm). The best pipelines merge these — ranking the merged
   evidence is where ordering gains hide.
5. **Concrete data traps to avoid (EDA):** precursor peak missing in 44% of test spectra (treat as
   metadata); 93% of test peaks are <1% base → the intensity floor is the biggest preprocessing dial;
   collision_energy_orig is NEGATIVE in neg-mode (use collision_energy_ev); median test spectrum 230
   raw peaks → ~18 real peaks after 1% floor.
6. **Noise discipline confirmed:** identical kernel resubmission moves score ±0.006 by seed alone.
   Any single-submission delta < ~0.012 is unproven.

## REVISED NEXT STEPS (ordered by EV)
- **A. Reverse-engineer the 0.396 leader.** One team is ~0.06 above the pack. If their approach is
  pool-instanceable, that's the single biggest jump toward us. (Check if Ozymandias published /
  their submission shape.) *Cheapest high-value action.*
- **B. Re-open Phase 4 with inversion's architecture (the real framing), not my tiny probe** — a
  768-dim spectrum→SMILES generator, validated honestly, only admitted into the final 25 if it
  clears an evidence gate. This is the "different framing" gate; +f₃ is the only path past the pack.
- **C. Squeeze ordering we can actually claim:** per-molecule multi-spectra evidence merge + better
  class priors + one genuinely-new input view — each gated SUBMISSION-side (local numbers distrust).
- **D. Do NOT** burn more submissions on fp-weight swaps, dedup, or pool expansion — all measured-dead.

## BOTTOM LINE
0.328 is a solid, honest medal-range lithology. The field is a plateau — **the only real upside left
is (A) reverse-engineering the outlier + (B) a serious de-novo generator.** Everything else is noise.