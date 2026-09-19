"""F-PHASE2 runner — F-RANKER + F-CLASS(class-prior calibration) + F-CONS
(consensus clustering), measured MRR@25 on the same SEED-11 60-mol holdout as
the Phase-1 incumbent (MRR 0.2670). Gate: must strictly beat 0.2670 to ship.

Run later by a background process:
    python src/phase2_runner.py

Prints `[F-PHASE2] MRR@25 = ...` and PASS/FAIL vs 0.2670, and persists the
upgraded model to data/franker_phase2.pkl. Same split/logic as f_ranker.py so
the number is directly comparable. No heavy compute in this file's control
flow beyond the incumbent training (reused verbatim via f_ranker._fit_persist).
"""
from __future__ import annotations
import os, sys
import numpy as np
import pyarrow.parquet as pq
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import f_ranker as FR
import f_cons
import f_class

MODEL_PATH = os.path.join(ROOT, "data", "franker_phase2.pkl")
INCUMBENT = 0.2670
SEED = 11
N_HOLDOUT = 60
N_TRAINQ = 120
CONS_THRESHOLD = 0.7
TOPN = 25


def _split(train_path, pool_key_set):
    """Recreate the exact Phase-1 SEED-11 split (same code path as f_ranker)."""
    rng = np.random.default_rng(SEED)
    t2 = pq.read_table(train_path, columns=["inchikey14"])
    df = t2.to_pandas().drop_duplicates("inchikey14")
    reach = [k for k in df.inchikey14.values if k in pool_key_set]
    eval_keys = set(rng.choice(reach, size=min(N_HOLDOUT, len(reach)), replace=False).tolist())
    remaining = [k for k in reach if k not in eval_keys]
    train_keys = remaining[:N_TRAINQ]
    return eval_keys, train_keys


def main():
    # 1) Train the Phase-1 rankers (exact incumbent path) and reuse its context.
    RANKERS, proba, ctx = FR._fit_persist()
    L = ctx["L"]; rep = ctx["rep"]; nmr = ctx["nmr"]
    key2pool = ctx["key2pool"]; pool_fp = ctx["pool_fp"]
    pool_keys = ctx["pool_keys"]; pool_mass = ctx["pool_mass"]

    # 2) Rebuild the identical split so train calibration rows match exactly.
    pool_key_set = set(pool_keys.tolist())
    eval_keys, train_keys = _split(FR.TRAIN, pool_key_set)
    assert eval_keys == ctx["eval_keys"], "f-phase2: split drifted from incumbent"

    # 3) F-CLASS: collect training rows (features + base proba + class) to fit
    #    the per-class empirical priors and isotonic recalibration.
    p_rows, y_rows, c_rows = [], [], []
    tset = set(train_keys)
    for r in rep:
        k = str(L.ik[r])
        if k not in tset:
            continue
        res = FR.query_features(L, key2pool, pool_fp, pool_keys, pool_mass, nmr, rep, r)
        if res is None:
            continue
        Xf, y, _, _, _ = res
        p_rows.append(proba(Xf)); y_rows.append(y)
        c_rows.append(f_class.class_of(Xf))
    if not p_rows:
        raise SystemExit("[F-PHASE2] no training rows for calibration!")
    P = np.concatenate(p_rows); Y = np.concatenate(y_rows); C = np.concatenate(c_rows)
    priors, calib = f_class.fit_recalibration(P, Y, C)
    n_pos = int(Y.sum()); n_tot = int(len(Y))
    print(f"[F-PHASE2] calibration set: {n_tot} rows, {n_pos} positives "
          f"({n_pos / max(1, n_tot):.4f}); class priors {priors}")

    # 4) Holdout eval: F-CLASS scores then F-CONS diverse top-k.
    mrr = 0.0; hit = 0; q = 0
    for r in rep:
        k = str(L.ik[r])
        if k not in eval_keys:
            continue
        res = FR.query_features(L, key2pool, pool_fp, pool_keys, pool_mass, nmr, rep, r)
        if res is None:
            continue
        Xf, y, _, cand, kk = res
        base = proba(Xf)
        classes = f_class.class_of(Xf)
        score = f_class.calibrate(base, priors, classes=classes, calib=calib)
        cid, _ = f_cons.cluster_candidates(pool_fp, cand, threshold=CONS_THRESHOLD)
        chosen = f_cons.diverse_rank(score, cid, topk=TOPN)
        rank = None
        for pos in chosen:
            if str(pool_keys[cand[pos]]) == kk:
                rank = pos + 1
                break
        q += 1
        if rank:
            mrr += 1.0 / rank
            hit += 1
    mrr /= max(1, q); h25 = hit / max(1, q)

    # 5) Report + persist the upgraded model.
    status = "PASS" if mrr > INCUMBENT else "FAIL"
    print("=" * 64)
    print(f"[F-PHASE2] MRR@25 = {mrr:.4f} | hit@25 = {h25:.4f} | n={q} "
          f"| gate> {INCUMBENT:.4f} (Phase-1 incumbent) -> {status}")
    print("=" * 64)
    payload = {"rankers": RANKERS, "feat_dim": FR.NFEAT,
               "priors": priors, "calib": calib, "cons_threshold": CONS_THRESHOLD,
               "mrr": mrr, "hit_at_25": h25}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(payload, MODEL_PATH)
    print(f"[F-PHASE2] persisted {MODEL_PATH} ({os.path.getsize(MODEL_PATH):,} bytes)")
    return mrr


if __name__ == "__main__":
    main()