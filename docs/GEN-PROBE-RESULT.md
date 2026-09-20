# CASMI26 — Phase-4 Generation Feasibility Probe: RESULT

**Date:** 2026-09-19  (kernel `aayanshabbir/casmi26-gen-probe-generation-feasibility`, GPU)
**Question:** For molecules with NO structure in any candidate pool/library (true Class-3),
can a model recover the EXACT structure (InChIKey14) from MS/MS spectra alone?

## Setup (leak-free)
- Train 60,000 / val 27,581 molecules, split BY STRUCTURE (inchikey14 dedup) — val structures
  NEVER seen in training = honest class-3 simulation.
- Small spectrum→SMILES transformer (10.5M params): spectrum binned to 0.05 Da over 1200 Da
  (24,001 bins) → dense MLP memory → 4-layer transformer decoder over SMILES chars (vocab 40).
- Greedy top-1 decoding, exact-match scored by RDKit tautomer-canonical InChIKey14.

## Result — NO-GO
| step | loss | val exact | val valid |
|---|---|---|---|
| 0 | 4.618 | 0/120 | 0 |
| 1000 | 0.869 | 0/120 | 0 |
| 2000 | 0.764 | 0/120 | 0 |
| 3000 | 0.700 | 0/120 | 0 |
| 3750 | 0.651 | 0/120 | 0 |

Model learns token-level SMILES statistics (loss falls to ~0.62) but never produces a valid,
let alone exact, structure for a held-out molecule. **0/120 exact, 0/120 valid at every step.**

## Interpretation
- Naive auto-regressive spectrum→SMILES **does not learn to generate held-out structures** in
  this framing within 4000 steps. Loss collapse ≠ structure recovery.
- This matches the field's dominant null hypothesis: exact structure generation from MS/MS for
  UNSEEN molecules is not achievable with a plain seq2seq. It is why nobody has a winning
  generator — the leaders all stay inside the ~0.43 retrieval ceiling.
- Caveat: 0 valid suggests the greedy decoder wasn't producing parseable strings even late in
  training — a decode/coverage deficiency at minimum; exact-recovery being exactly 0 holds
  regardless. A much larger model / beam-search / pretrained-ChemBERTa decoder COULD do better,
  but the cheap probe shows the naive path is not worth weeks of GPU.

## Decision
**Do NOT invest GPU weeks on the naive spectrum→SMILES generator.**
- Generation (Phase 4) is parked as LOW-priority — reopen only if a fundamentally different framing
  (e.g. fingerprint-predicted→conditional graph diffusion, or a strong pretrained ChemBerta-aligned
  decoder) shows signal in ITS OWN probe first.
- Redirect effort to **Track A: ordering/ranking polish** on the working 0.328 retrieval engine
  (better per-spectrum fp model, second input view, seeds) → target 0.34–0.35, the realistic
  ceiling with this pool.

## Files
- Probe notebook: `/tmp/probe/gen_probe.ipynb` (also `casmi26-gen-probe-generation-feasibility` v5, Kaggle).
- Full log: verbose per-step loss + exact/valid (see above table).