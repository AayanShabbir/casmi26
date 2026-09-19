"""F-SUBMIT — end-to-end CASMI26 inference + submission generation.

Loads the persisted F-RANKER model (data/franker_model.pkl), builds the
train-analog rep library + trainpool candidate library, and for each of the
400 unique test molecules runs the SAME combined-evidence feature extraction
as query_features, scores candidates with the 8-rankers mean proba, ranks,
takes top-25, and maps to SMILES. Emits submission.csv in sample_submission
format (molecule_id,smiles with 25 ";"-joined SMILES).

No heavy compute is expected here beyond the per-query feature pass; the model
must already be trained + persisted by f_ranker.train_and_persist().
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pyarrow.parquet as pq
import joblib
from rdkit import Chem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import analog_accel as A
import f_ranker as FR

TRAIN = os.path.join(ROOT, "train.parquet")
TEST = os.path.join(ROOT, "test.parquet")
POOL_CACHE = os.path.join(ROOT, "data", "trainpool.npz")
MODEL_PATH = os.path.join(ROOT, "data", "franker_model.pkl")
OUT_CSV = os.path.join(ROOT, "submission.csv")
TOPN = 25


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"[F-SUBMIT] model missing at {MODEL_PATH}\n"
            "  Run `python src/f_ranker.py` (or f_ranker.train_and_persist()) to "
            "train + persist the 8 rankers first.")
    payload = joblib.load(MODEL_PATH)
    rankers = payload["rankers"]
    feat_dim = payload.get("feat_dim", FR.NFEAT)
    if feat_dim != FR.NFEAT:
        raise SystemExit(
            f"[F-SUBMIT] model feat_dim={feat_dim} != expected {FR.NFEAT}; "
            "feature pipeline mismatch. Retrain.")
    def proba(X):
        return np.mean([m.predict_proba(X)[:, 1] for m in rankers], axis=0)
    return rankers, proba


def build_key_to_smiles():
    t = pq.read_table(TRAIN, columns=["inchikey14", "normalized_smiles"])
    df = t.to_pandas().dropna(subset=["inchikey14"])
    df = df.drop_duplicates(subset=["inchikey14"], keep="first")
    return dict(zip(df["inchikey14"].astype(str), df["normalized_smiles"].astype(str)))


def build_test_lib():
    """Lib-like structure over test.parquet keyed by molecule_id. Reuses the
    exact analog_accel cleaning path (attributes off/mz/it/n + _clean_all)."""
    t = pq.read_table(TEST, columns=["molecule_id", "adduct", "precursor_mz",
                                     "ms2_mzs", "ms2_normalized_intensities"])
    ik = np.asarray(t.column("molecule_id").cast("string").to_pylist(), dtype=object)
    add = np.asarray(t.column("adduct").cast("string").to_pylist(), dtype=object)
    prec = t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64)
    mzc = t.column("ms2_mzs").combine_chunks()
    itc = t.column("ms2_normalized_intensities").combine_chunks()
    L = object.__new__(A.Lib)          # skip __init__ (reads train schema)
    L.ik = ik
    L.nm = A.neutral_mass(prec, add)
    L.off = mzc.offsets.to_numpy().astype(np.int64)
    L.mz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    L.it = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    L.n = len(ik)
    L._clean_all()                     # identical cleaning path -> cmz/cp/coff
    return L


def pick_reps(L):
    """One representative row per molecule_id: first finite-neutral-mass row,
    identical selection rule to f_ranker's train rep building."""
    best = {}
    for r in range(L.n):
        k = L.ik[r]
        if k and np.isfinite(L.nm[r]) and k not in best:
            best[k] = r
    rep = np.array(sorted(best.values())); nmr = L.nm[rep]; oor = np.argsort(nmr)
    return rep[oor], nmr[oor], best


def main():
    t0 = time.time()
    rankers, proba = load_model()
    print(f"[model] {len(rankers)} rankers @ {MODEL_PATH} {time.time()-t0:.0f}s", flush=True)

    key2smiles = build_key_to_smiles()
    print(f"[map] {len(key2smiles)} inchikey14->SMILES {time.time()-t0:.0f}s", flush=True)

    TL = A.Lib(TRAIN)                  # train analog library (reps + cleaned spectra)
    pool_fp, pool_mass, pool_keys = FR.load_pool()
    key2pool = {k: i for i, k in enumerate(pool_keys.tolist())}
    print(f"[pool] {len(pool_mass):,} candidates {time.time()-t0:.0f}s", flush=True)

    # train analog reps: first row per train inchikey, sorted by neutral mass
    best = {}
    for r in range(TL.n):
        k = TL.ik[r]
        if k and np.isfinite(TL.nm[r]) and k not in best:
            best[k] = r
    trep = np.array(sorted(best.values())); tnmr = TL.nm[trep]; o = np.argsort(tnmr)
    trep = trep[o]; tnmr = tnmr[o]
    print(f"[analog-reps] {len(trep)} train reps {time.time()-t0:.0f}s", flush=True)

    QT = build_test_lib()              # test lib keyed by molecule_id
    trep_i, tnmr_i, best_m = pick_reps(QT)
    nmols = len(best_m)
    print(f"[test-reps] {nmols} molecules {time.time()-t0:.0f}s", flush=True)

    analog_args = dict(rep=trep, nmr=tnmr, acoff=TL.coff, acmz=TL.cmz,
                       acp=TL.cp, aik=TL.ik, key2pool=key2pool,
                       pool_fp=pool_fp, pool_keys=pool_keys, pool_mass=pool_mass)

    rows = []
    missing = 0
    for r in trep_i:
        qid = str(QT.ik[r]); qtarget = float(QT.nm[r])
        a, b = QT.off[r], QT.off[r + 1]
        qm, qp = QT.cmz[a:b], QT.cp[a:b]
        res = FR.features_for_query(qm, qp, qtarget, **analog_args, qkey=qid)
        if res is None:
            missing += 1
            rows.append((qid, []))
            continue
        Xf, y, _, cand, _ = res
        p = proba(Xf)
        order = np.argsort(-p)
        keys = [str(pool_keys[cand[j]]) for j in order[:TOPN]]
        rows.append((qid, keys))
        if len(rows) % 50 == 0:
            print(f"  [{time.time()-t0:.0f}s] {len(rows)}/{nmols} processed", flush=True)

    # map to SMILES; pad short candidate lists with the top analog's SMILES
    written = []
    pad_cache = []
    for qid, keys in rows:
        smis = []
        for k in keys:
            s = key2smiles.get(k)
            if s:
                smis.append(s)
        if not smis and pad_cache:
            smis = [pad_cache[0]] * TOPN
        elif len(smis) < TOPN and pad_cache:
            smis = smis + [pad_cache[0]] * (TOPN - len(smis))
        if not smis:                    # no candidate mapped and no pad source yet
            smis = ["CCO"] * TOPN
        while len(smis) < TOPN:
            smis.append(pad_cache[0])
        if smis:
            pad_cache = smis
        written.append((qid, ";".join(smis[:TOPN])))

    with open(OUT_CSV, "w") as f:
        f.write("molecule_id,smiles\n")
        for qid, smi in written:
            f.write(f"{qid},{smi}\n")
    print(f"[F-SUBMIT] wrote submission.csv ({len(written)} rows) "
          f"| missing_features={missing} | {time.time()-t0:.0f}s", flush=True)
    return OUT_CSV, len(written)


if __name__ == "__main__":
    main()