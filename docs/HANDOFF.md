# HANDOFF — CASMI26 (read first in a fresh session)

Repo: `~/code/casmi26` — standalone git repo (own `.git`, own files, data gitignored).
Ven v: `~/code/casmi26/.venv` (rdkit 2025.09.2, numba 0.60, numpy, pyarrow, sklearn, joblib).
Kaggle CLI `~/.local/bin/kaggle` 2.2.4 (auth via ~/.kaggle/kaggle.json, user `aayanshabbir`).

## GOAL
Top-10% medal (MRR@25 ~0.30+) on Enveda CASMI26 (400 mol, deadline 2026-12-14). Winner ~0.339.
Competition is a **CODE competition — submissions are notebook-only** (CLI file-upload 400s by design;
notebooks must be **fully offline** to be submittable).

## MEASURED STATE (all committed & verified)
- **F-ANALOG baseline:** MRR 0.2125 / hit 62.75% (train-pool holdout, SEED 11, 51-60 mol).
- **F-SPEED kernel:** `f_speed.search_shift_prange` prange-parallel, **4.3ms/3000 reps**, bit-exact vs serial.
  (HANDOFF correction: serial was never a real wall — 6µs/rep; the ">120s" premise was a phantom.)
- **F-RANKER (Phase 1, CLOSED):** **MRR 0.2670**, hit 64.7%, on the SAME SEED-11 holdout.
  8× HistGB (2prior×4seed), **20 features** (`features_for_query`), F-SPEED swap in.
  Leakage audit **CLEAN** (train-positive-taint check; the earlier "LEAKED" was my audit's bad assertion).
- **Stage artifacts:** `data/franker_model.pkl` (6.2MB) = {"rankers":8×HistGB,"feat_dim":20}.
  `data/trainpool.npz` (96MB) = {fp:(275,810,1280) uint8 bytepacked, mass, keys} — used for train AND test.

## SUBMISSION STATUS ⚠️ (the live thread to finish)
- Kernel `aayanshabbir/casmi26-analog-boosted-ranker` on Kaggle, **version 8, COMPLETE (no error)**,
  fully-offline. Produced `submission.csv` (401 lines = header + 400, 0 invalid, all 25-padded).
- Comp submission **ref 56368312, "First", status PENDING at last check (~19:00 EDT)** — was scoring.
  **FIRST thing in a fresh session: poll `kaggle competitions submissions -c enveda-CASMI26-molecule-id-mass-spectra`
  and read the publicScore.** Local copy: `kaggle/submission.csv`.
- Expected ~0.24–0.28 (matches holdout 0.2670; pool coverage solid: only 1/400 mols lacked features).

## THE OFFLINE-SUBMITTABLE RECIPE (the hard-won part — do not regress)
Mounted input lives at **`/kaggle/input/competitions/<id>/`** (NOT `/kaggle/input/<id>/`). So:
1. **Input path:** resolve via `os.walk('/kaggle/input')` for `train.parquet`/`test.parquet` (recursive, robust).
2. **Pool:** shipped as **private dataset `aayanshabbir/casmi26-trainpool`** (92MB, trainpool.npz)
   -> loaded in ~1s via np.load, NO rdkit needed.
3. **No pip / no rdkit / `enable_internet: false`** — this is what makes it submittable. The ONLY external
   deps are numpy/pyarrow/sklearn/numba (base image). rdkit is used ONLY for pool build (done once locally).
4. **Adduct map must cover all 11 test adducts** incl negatives: `[M-H]-`, `[M+CH2O2-H]-`, `[M-H2O-H]-`,
   `[M+K]+`, `[M+Cl]-`, `[M-2H2O+H]+`, `[M-H2O+H]+`. Missing them silently drops molecules (27 in 1st run).
5. **Local-pretest before pushing:** extract ipynb code cells -> run as a script against a local mirror of
   the /kaggle/input layout -> confirm 400-row output BEFORE `kaggle kernels push`. This killed the
   error-push-error-push cycle (numba import, t0 ordering, path glob all caught locally first).
6. Kernel metadata: `enable_gpu: false`, `enable_internet: false`, `competition_sources` + dataset_sources.

## FLEET / EXECUTION REALITY (anti-patterns, learned the hard way)
- **Every long background process gets SIGTERM'd by the Hermes runtime** (`agent_close`) after a while.
  Code-writing fleet agent runs **finish**; heavy-compute runs (train/infer) **die** unless actively
  `wait`ed on. **Pattern that works: launch background=true then IMMEDIATELY chain process wait(180s)
  calls** so the session never idles. Do NOT `-Q -q` (hides wedges) and do NOT shell-`&`-detach (SIGKILL 0-byte).
- Fleet AGY lane (`agy-bridge` gemini-3.8 @ :8790) works for code-writing; architect/routes visibly.
- Persist + submit runs all worked fine ~21 min wall when manually waited.

## NEXT PHASES (architect plan `docs/ARCHITECT_PLAN.md`, committed — each gated > incumbent)
- **F-CLASS & F-CONS (Phase 2)** — code ALREADY written by researcher: `src/f_cons.py`, `src/f_class.py`,
  `src/phase2_runner.py` (Platt/prior calibration + consensus cluster diversity). **Never run yet** (blocked
  by SIGTERM kills earlier). Run `python src/phase2_runner.py` locally, gate > 0.2670, then new submission.
- Then F-FRAG/F-MASS, F-FP/SPEC2VEC (two-ranker leak-safe blend), F-ENSEMBLE/SUBMIT per arch plan.

## NOTES
- Dispatched briefs (lean-pointer, AGY): `fleet_brief_*.txt` all in repo root. Committed.
- Do NOT touch `~/dev/papertrade/` (Alpaca day-trading risk structures — protected).
- Cheaper to iterate locally (fast) and only push to Kaggle when locally green.