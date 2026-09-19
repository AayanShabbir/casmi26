# CASMI26 Master Architectural Roadmap & Fleet Plan

**Document:** `~/code/casmi26/docs/ARCHITECT_PLAN.md`  
**Status:** Authoritative Architectural Directive  
**Date:** 2026-09-18  
**Author:** Fleet Architect (`persona.architect`, AGY lane :8790)  
**Savant Workspace:** `CASMI26 Molecule ID` (Workspace ID: `2277616941600993050`)  
**Target Objective:** Top-10% CASMI26 medal (MRR@25 >= 0.300, winner baseline 0.339), Deadline 2026-12-14.  
**Core Baseline:** F-ANALOG incumbent measured MRR@25 = **0.2125** (Hit@25 = 62.75% on 51-molecule reachable holdout).  
**F-SPEED Status:** Complete (`src/f_speed.py`, 4.3 ms/batch, 4.1x bit-exact). Kernel is not a bottleneck.

---

## 1. Roadmap Phases & Gated Milestones

The CASMI26 problem space is strictly candidate ranking (~132 mass-matched structures within +-10 ppm per query), not novel discovery. Every phase is an incremental evidence channel grafted onto candidate ranking. The gate law is absolute: **every phase must beat the incumbent measured MRR@25 on the standard held-out validation split before promoting to master**.

```
[F-ANALOG (0.2125)] --> [Phase 1: F-RANKER (>0.233)] --> [Phase 2: F-CLASS/F-CONS (>0.245)]
                                                                    |
[Phase 5: F-ENSEMBLE/SUBMIT (>=0.335)] <-- [Phase 4: F-FP/SPEC2VEC (>0.299)] <-- [Phase 3: F-FRAG/F-MASS (>0.266)]
```

### Phase 1: F-RANKER — 31-Feature GBDT Candidate Ranker
- **Goal:** Replace heuristic `sim^4 * Tanimoto` scoring with a trained `HistGradientBoosting` model ranking candidates from pooled evidence (analog similarity, library match, precursor mass deviation, popcount density, candidate cohort size).
- **Exit Gate:** Holdout validation MRR@25 **> 0.2330** (incumbent: 0.2125). Zero regression on Hit@25 (>= 62.75%).
- **Dependencies:** `src/f_ranker.py`, `data/trainpool.npz`, `src/f_speed.py`.
- **Savant KG / Workspace Node:** `task.casmi26.f_ranker`, KG node `service:casmi26-ranker` under domain `domain:casmi26`.

### Phase 2: F-CONS & F-CLASS — Consensus Calibration & Class-Prior Weighting
- **Goal:** Integrate class-conditioned weight calibration (W1=0.30 vs 0.60 for Class-1 direct match vs Class-2 analog match) and consensus voting across analog clusters to suppress false positive analog anchors.
- **Exit Gate:** Holdout validation MRR@25 **> 0.2450** (incumbent: Phase 1 output >= 0.2330).
- **Dependencies:** Phase 1 ranker model artifact (`data/f_ranker_v1.pkl`), `src/f_consensus.py`.
- **Savant KG / Workspace Node:** `task.casmi26.f_class_consensus`, KG node `insight:class-prior-weighting`.

### Phase 3: F-FRAG & F-MASS — In-Silico Substructure & Mass-Shift Alignment
- **Goal:** Adduct-aware substructure bond-break scoring (MetFrag-lite logic) evaluating MS2 fragment match against candidate smiles, paired with neutral loss mass-shift alignment (`F-MASS`).
- **Exit Gate:** Holdout validation MRR@25 **> 0.2660** (incumbent: Phase 2 output >= 0.2450).
- **Dependencies:** RDKit fragment generator, `src/f_frag.py`, `src/f_mass.py`.
- **Savant KG / Workspace Node:** `task.casmi26.f_frag_mass`, KG node `technology:insilico-fragmentation`.

### Phase 4: F-FP & F-SPEC2VEC — Neural Fingerprint & Spec2Vec Representation
- **Goal:** Embed spectra via pretrained Spec2Vec and public FPNet fingerprint predictions. Crucial requirement: **Two-Ranker leak-safe architecture** (leak-free ranker + leaky ranker blended 0.65/0.35 to eliminate training structure memorization).
- **Exit Gate:** Holdout validation MRR@25 **> 0.2990** (incumbent: Phase 3 output >= 0.2660).
- **Dependencies:** Pretrained weights, `src/f_fpnet.py`, `src/f_spec2vec.py`.
- **Savant KG / Workspace Node:** `task.casmi26.f_fp_spec2vec`, KG node `technology:two-ranker-mitigation`.

### Phase 5: F-ENSEMBLE & F-SUBMIT — Seed Bagging & Kaggle Submission Packaging
- **Goal:** Merge 4-seed GBDT models with dual-prior weights into an ensemble prediction pipeline. Package inference engine into Kaggle-compliant standalone script, generate `submission.csv`, verify SHA256 and bit-exact reproducibility.
- **Exit Gate:** Standalone local evaluation MRR@25 **>= 0.3350**; submission format verification passes `sample_submission.csv` schema validation (400 queries x 25 ranked InChIKey14s, 100% valid SMILES/keys).
- **Dependencies:** All prior model artifacts, `test.parquet`, `sample_submission.csv`.
- **Savant KG / Workspace Node:** `task.casmi26.f_submission`, KG node `operation:casmi26-final-submit`.

---

## 2. Fleet Agent Assignment Matrix

To prevent single-agent wedge failures, **EVERY phase is executed by exactly TWO distinct agents** with complementary specializations. 

Total fleet agents involved across the entire deal: **6 distinct agents** drawn from the 13-persona fleet:
1. `engineer` (Systems implementation, high-throughput pipelines, numerical acceleration)
2. `checker` (Adversarial gatekeeper, leak verification, metric auditing)
3. `programmer` (Algorithm implementation, feature engineering, statistical models)
4. `researcher` (Literature mapping, spectral representation, prior weighting)
5. `solver` (Heuristics, combinatorial optimization, graph traversal)
6. `tester` (Harness validation, regression tracking, packaging compliance)

### Per-Phase Agent Pairs

| Phase | Primary Agent | Secondary Agent | Rationale for Pairing |
| :--- | :--- | :--- | :--- |
| **Phase 1: F-RANKER** | `programmer` | `checker` | `programmer` tunes feature matrix and GBDT; `checker` audits candidate leakage and validates true holdout MRR > 0.2330. |
| **Phase 2: F-CLASS / F-CONS** | `researcher` | `solver` | `researcher` defines class probability distributions; `solver` implements consensus cluster aggregation and weight calibration. |
| **Phase 3: F-FRAG / F-MASS** | `engineer` | `programmer` | `engineer` optimizes fast substructure bond dissociation; `programmer` maps neutral loss alignment features into ranking vectors. |
| **Phase 4: F-FP / SPEC2VEC** | `engineer` | `checker` | `engineer` integrates Spec2Vec / FPNet tensors; `checker` enforces the Two-Ranker split (0.65/0.35) to prevent library target leak. |
| **Phase 5: F-ENSEMBLE / SUBMIT**| `tester` | `checker` | `tester` verifies end-to-end execution on `test.parquet` and reproducible seed bagging; `checker` validates schema, Kaggle limits, and final submission integrity. |

**Total Unique Agents:** 6  
**Phase Execution Redundancy:** 2 agents per phase (100% dual-crewed). Zero single points of failure.

---

## 3. Disjoint Slicing & File Isolation Rules

Agents within a pair work on strictly partitioned files and duties to eliminate file write conflicts, merge contention, and git collisions.

```
+--------------------------------------------------------------------------+
|                       DISJOINT SLICING ARCHITECTURE                      |
+------------------------------------+-------------------------------------+
| SLICE A: Feature / Engine Producer | SLICE B: Harness / Metric Consumer   |
| - Writes implementation to src/    | - Writes verification to bench/     |
| - Emits raw feature / data npz     | - Consumes npz, runs holdout audit  |
| - Optimizes runtime throughput     | - Audits leakage & computes MRR@25  |
+------------------------------------+-------------------------------------+
```

### Explicit Slicing Boundaries

1. **Phase 1 (F-RANKER):**
   - `programmer` owns `src/f_ranker.py` and `data/rank_features.npz`. Implements 31 feature columns, trains GBDT model.
   - `checker` owns `bench/eval_ranker.py` and `docs/RANKER_AUDIT.md`. Validates label integrity (zero candidate contamination), computes strict holdout MRR@25.

2. **Phase 2 (F-CLASS / F-CONS):**
   - `researcher` owns `src/f_priors.py` and calibration metadata `data/class_priors.json`. Defines W1/W2 class-conditioned distributions.
   - `solver` owns `src/f_consensus.py` and `bench/eval_consensus.py`. Builds graph-based analog cluster consensus and benchmarks combined score.

3. **Phase 3 (F-FRAG / F-MASS):**
   - `engineer` owns `src/f_frag_kernel.py` and `src/f_frag.py`. High-speed fragment generation and peak matching.
   - `programmer` owns `src/f_mass.py` and `src/f_feature_merge.py`. Neutral-loss spectral alignment and feature matrix expansion.

4. **Phase 4 (F-FP / SPEC2VEC):**
   - `engineer` owns `src/f_spec2vec.py` and embedding cache `data/spec2vec_embeddings.npz`.
   - `checker` owns `src/f_two_ranker.py` and `docs/LEAK_AUDIT.md`. Enforces leak-free training partitions and 0.65/0.35 blending weights.

5. **Phase 5 (F-ENSEMBLE / SUBMIT):**
   - `tester` owns `src/ensemble_pipeline.py`, `src/predict_submission.py`, and `submission.csv`.
   - `checker` owns `bench/verify_submission.py` and `docs/SUBMISSION_VERIFICATION.md`. Runs independent checksum, format, and range verification.

---

## 4. Lane, Runtime & Execution Bounds

To prevent the known fleet wedge failure (where local agent profiles spawn heavy local runtimes and wedge at 0% CPU), the runtime rules are non-negotiable:

1. **Bridge Protocol:** All fleet agents operate exclusively over the AGY bridge (`:8790`, model `gemini-3.8-flash-medium`). OpenRouter and direct paid reasoning tiers are strictly prohibited for phase execution.
2. **Turn Budgeting:**
   - Max turns per agent per phase: **15 turns**.
   - Max run-budget per agent per turn: **120 seconds** timeout on shell/tool execution.
3. **Heavy Compute Execution Law:** Heavy numerical and training compute (> 30s) must **never** be executed interactively inside agent chat loops. Heavy workloads must be dispatched as standalone background scripts with progress logging to disk (e.g., `python src/train_ranker.py > logs/ranker.log 2>&1 &`), monitored via lean poll checks.

---

## 5. Handoff Contract Between Phases

Every phase handoff occurs through persistent, versioned disk artifacts. Direct in-memory payload passing between agents or phases is prohibited.

### Required Handoff Bundle per Phase

Every phase must emit the following 4 files into `docs/` and `data/` before the next phase may consume its output:

1. **Model / Logic Artifact:** `data/<phase_name>_model.npz` or `data/<phase_name>_weights.pkl`.
2. **Prediction Parquet:** `data/<phase_name>_holdout_preds.parquet` (columns: `query_id`, `inchikey14`, `rank`, `score`).
3. **Standard Metric Receipt:** `docs/<phase_name>_RECEIPT.json`:
   ```json
   {
     "phase": "F-RANKER",
     "incumbent_mrr": 0.2125,
     "measured_mrr": 0.2348,
     "hit_at_25": 0.6392,
     "status": "PASS",
     "git_commit": "<hash>",
     "timestamp": "2026-09-18T..."
   }
   ```
4. **Audit Report:** `docs/<phase_name>_REPORT.md` written by the secondary agent (`checker` / `tester`) verifying lack of data leakage, reproducibility, and exit code 0.

---

## 6. Critical Success Factors & Risk Mitigation

| Identified Risk | Severity | Root Cause | Architectural Mitigation |
| :--- | :--- | :--- | :--- |
| **Fleet Session Wedge** | Critical | Fleet profiles attempting heavy MS2 search or GBDT training in interactive chat wedge at 0% CPU. | Disjoint pairing + background scripts with disk logging. Agents write code, launch headless runs, and evaluate outputs via lean status checks. |
| **Data & Target Leakage** | Critical | FPNet / Spec2Vec weights trained on CASMI candidate structures give artificial 0.80+ validation MRR. | Phase 4 mandates Two-Rankers architecture (0.65 leak-free / 0.35 leaky blend) audited by `checker` before promotion. |
| **COCONUT Pool Divergence** | High | Using synthetic vs natural product pools yields 0% overlap with target holdouts. | Retain strictly `data/trainpool.npz` (275,810 structures) with verified self-consistent Morgan2+3+RDKit fingerprints. |
| **File Collisions** | Medium | Multiple agents simultaneously modifying shared ranker script. | Strict disjoint file ownership (Slice A: `src/f_*.py` vs Slice B: `bench/eval_*.py`). |
| **Phantom Compute Bottleneck** | Medium | Optimizing kernel when bottleneck is Python orchestration. | F-SPEED already confirmed serial kernel is 4.3 ms. Focus compute budget on vectorization of `rep->agg` and Tanimoto windowing. |

---

## 7. Immediate Activation Directive

1. **Active Phase:** Phase 1 (`F-RANKER`).
2. **Lead Agent:** `programmer` (owns `src/f_ranker.py`).
3. **Secondary Agent:** `checker` (owns `bench/eval_ranker.py` and gate signoff).
4. **Incumbent Target:** Holdout MRR@25 **> 0.2125**.
5. **First Milestone Deliverable:** `docs/F_RANKER_RECEIPT.json` demonstrating verified improvement.
