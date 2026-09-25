# CASMI26 — analog-backed molecular identification for Kaggle Enveda CASMI26

Rank the **correct molecule first** among a mass window of candidates for unknown
MS/MS spectra — competitive molecular identification from mass spectrometry.

**Task:** For 400 query spectra, return a ranked list of up to 25 SMILES
(`MRR@25`). The hard part is not retrieval (a matched library handles ~100% of
real molecules); it is picking the **right** structure out of the ~132
mass-window candidates. That makes this a *ranking* problem.

Target: top-10% medal (~0.30+ MRR), reproducing the public winner ladder
(0.335–0.339) by grafting stronger open-source channels.

> Status: competitive project, tracked in the `PLAN.md` living plan with
> measured exit gates per stage. Deadline 2026-12-14.

## Approach

A combined-evidence pipeline built in stages, each gated on measured MRR
against the incumbent (no swap ships on a hunch):

| Stage | Method | Measured (held-out) |
|---|---|---|
| **F-BASE** | Entropy-weighted spectral cleaning + cosine library retrieval in a tight neutral-mass window | self-exclude MRR@25 **0.704** / hit@25 0.843 |
| **F-ANALOG** | Mass-shifted analog propagation: ±200 Da window, `sim(a)^4 · Tanimoto(fpᶜ, fpᵃ)`, candidate pool of ~711k (COCONUT ∪ train) | single-channel MRR@25 **0.212** |
| **F-RANKER** | 20-feature gradient-boosted candidate ranker (library sim, analog sim/rank, Tanimoto, counts, ppm error) with 8-rankers mean proba (2 class priors × 4 seeds) | MRR@25 **0.267** — beats F-ANALOG gate |
| **F-SUBMIT** | End-to-end inference → `submission.csv` (25 `;`-joined SMILES per molecule) | — |
| **F-CLASS / F-FRAG / F-FP** | Class-weight calibration, in-silico fragmentation (MetFrag-lite), leak-safe fingerprint channel (two-ranker blend) | ladder to ~0.335 |

### Why the two-ranker blend matters (leak control)

Public FPNet weights were trained on most library structures, so a ranker built
on them over-trusts the fingerprint channel (0.76–0.82 on seen sets vs 0.49
held-out). The winning mitigation — validated here — is **two rankers**
(leak-free + leak-exposed) blended ~0.65/0.35, with a strict split-integrity
audit (`bench/audit_ranker.py`) asserting the held-out and training keys are
disjoint before the MRR is trusted.

## Repo layout

```
src/
  analog.py          F-ANALOG core: entropy sim, mass-shift, Tanimoto (pure numpy)
  analog_accel.py    Accelerated core: njit spectral cleaning + mass-shift sims
  lib.py             F-BASE: cleaning + CSR-like sparse spectral index
  f_ranker.py        F-RANKER: 20-feature GBDT ranker, 8-model ensemble, MRR gate
  f_submit.py        F-SUBMIT: end-to-end inference -> submission.csv
  f_cons.py f_speed.py f_class.py  supporting stages
  run.py             F-BASE held-out runner (seed 0, MRR@25 / hit@25)
bench/
  audit_ranker.py    CHECKER: split-integrity + leakage audit for the ranker gate
kaggle/
  casmi26_kaggle.py, casmi26_notebook.ipynb, submissions/
  pool-dataset/      packaged candidate-pool dataset metadata
docs/                per-stage reports + measured results (f*_measured.txt)
```

## Dependencies

- Python 3.10+
- `numpy`, `numba`, `pyarrow`, `rdkit`, `scikit-learn`, `joblib`

Data files (`train.parquet`, `test.parquet`, `sample_submission.csv`) and the
candidate pool (`data/trainpool.npz`) are gitignored; obtain them from the
competition and build the pool from the public COCONUT dataset.

## Quick start

```bash
# 1. F-ANALOG held-out gate (single-channel analog)
python src/analog_run*.py

# 2. Train + persist the ranker, then measure the MRR@25 gate
python src/f_ranker.py          # trains 8 HistGB rankers -> data/franker_model.pkl
                                #   and measures MRR@25 vs the F-ANALOG incumbent

# 3. End-to-end inference -> submission.csv
python src/f_submit.py

# 4. Independent leakage / split-integrity audit of the ranker gate
python bench/audit_ranker.py
```

## Gate discipline

Every stage ships only when it beats the incumbent measured MRR head-to-head:
- F-ANALOG must reproduce ≥~0.50 single-channel on a fair Class-2 held-out
  (winner-proven achievable: 0.521/0.516).
- F-RANKER must beat the F-ANALOG incumbent on the SEED-11 holdout (audited
  disjoint split).
- F-FP must blend leak-free + leaky rankers; leak management is the #1
  correctness gate, not optional.

Measured stage results live in `docs/f*_measured.txt`; the full plan, findings,
and honest open risks are in `PLAN.md`.

## Author

Aayan Shabbir — Biochemistry major at Binghamton University, self-taught ML.

MIT licensed.