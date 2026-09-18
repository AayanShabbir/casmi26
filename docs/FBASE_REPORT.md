# CASMI26 F-BASE Report

Phase: F-BASE (fleet engineer + checker) — assemble one runner, reproduce the
measured Ch1 retrieval floor (MRR@25 >= 0.108), and lock the facility the later
phases (F-CONS, F-MASS, F-ALIGN, F-LEARN, F-ANALOG) build on.

## Runner

Absolute path: `/Users/aayan/code/casmi26/src/run.py`

Library module: `/Users/aayan/code/casmi26/src/lib.py`

Verify command:
```
.venv/bin/python src/run.py
```

## Inputs / data layout

- `train.parquet` (3.03 GB): CASMI26 training spectra — 2,539,608 spectra,
  275,810 unique `inchikey14` molecules, precursor m/z 2.0–1986.1. Columns used:
  `inchikey14` (label), `precursor_mz`, `ms2_mzs`, `ms2_normalized_intensities`.
  Intensities pre-normalized to max=1. Already present in worktree; not re-downloaded.
- `test.parquet` (4.85 MB), `sample_submission.csv`, `winner_submission.csv` — present,
  unused by F-BASE.

`data/` (COCONUT pool, from `kaggle datasets download -d prvsiyan/coconut-casmi26-candidates`):

| file | contents | shape |
| :--- | :--- | :--- |
| `coco_fp.npy` | RDKit 6930-bit fingerprints (sparse 867 active cols, uint8) | 436,389 × 867 |
| `coco_mass.npy` | neutral monoisotopic mass | 436,389 |
| `coco_meta.pkl` | dict: `keys` (inchikey14), `smiles`, `nbits`=6930 | 436,389 |
| `fp_bits.npy` | fingerprinted bit indices | 6,930 |

All 436,389 COCONUT inchikey14 are unique. `fp_bits` spans bit indices 2–10405
(active subset of the 6930-bit space).

## Recipe (F-BASE)

Per spectrum, from `train.parquet`:
1. Entropy-weighted peak cleaning:
   - intensity floor 0.002,
   - keep top 256 peaks by intensity,
   - +200 Da window: retain peaks with m/z <= precursor_mz + 200,
   - entropy sharpening (Li & Fiehn SpectralEntropy: if spectral entropy S < 3,
     intensity power = 0.25 + 0.25*S, then renormalize).
2. Neutral mass = precursor_mz - 1.00728.
3. Binned into 0.1 Da vectors (m/z 0–2000 -> 20,001 bins), L2-normalized,
   stored as CSR-like sparse index (numpy int32 indices / float32 values).

Retrieval (per held-out query):
- Candidate window = library spectra whose NEUTRAL mass is within +-8.5 ppm
  of the query's NEUTRAL mass (PPm on neutral, not precursor).
- Cosine rank candidate spectra; MRR@25 / hit@25 by `inchikey14`.
- The query's own row and exact twins (same inchikey14 + rounded precursor)
  are excluded from the ranked candidate set to avoid trivial self-match.

## Holdout

400 distinct train molecules (unique `inchikey14`), one query spectrum each,
selected with `numpy.random.default_rng(0)` (seed 0).

## Measured result

Command: `cd ~/code/casmi26 && .venv/bin/python src/run.py`

```
[F-BASE] index built: 2539608 spectra, nnz=...
[F-BASE] 400 holdout queries, querying ...
========================================================
[F-BASE] holdout  | MRR@25 = <RESULT> | hit@25 = <RESULT> (<PCT>%)
[F-BASE] gate     | target >= 0.108  | <PASS|FAIL>
========================================================
```

- **MRR@25 (measured):** PLACEHOLDER
- **hit@25 (measured):** PLACEHOLDER

## Gate

- Target floor: MRR@25 >= 0.108 (measured Ch1 floor from PLAN.md).
- Status: PLACEHOLDER

## Not wired (by design, later phases)

- COCONUT pool + RDKit fingerprints are present in `data/` but NOT yet used for
  ranking. `rank_train.npz` / FPNet are deferred to later phases.
- Fine m/z alignment (0.01 Da) is F-ALIGN, not F-BASE.