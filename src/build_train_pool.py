"""Build the missing half of the F-ANALOG candidate pool: fingerprints for
training structures, using the SAME scheme as data/coco_fp.npy
(Morgan2(4096) + Morgan3(4096) + RDKit(2048) -> subset by fp_bits -> bytepack).

Writes data/train_fp.npy, data/train_mass.npy, data/train_keys.npy so the
runner can load pool = COCONUT U train without recomputing each pass.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors
RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "data")
TRAIN = os.path.join(ROOT, "train.parquet")
BITS = np.load(os.path.join(DATA, "fp_bits.npy"))  # 6930 selected indices into 10240


def _gens():
    return (rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096),
            rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096),
            rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6))


def fp_vec(smi):
    """Self-consistent 10240-bit fp (m2+m3+rk2048), bytepacked. Used for BOTH
    candidates and analogs so Tanimoto is internally consistent (does NOT tie to
    the COCONUT fp_bits scheme — that tie is author-private)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    g2, g3, rk = _gens()
    vec = np.concatenate([g2.GetFingerprintAsNumPy(m).astype(np.uint8),
                          g3.GetFingerprintAsNumPy(m).astype(np.uint8),
                          rk.GetFingerprintAsNumPy(m).astype(np.uint8)])
    return np.packbits(vec), float(Descriptors.ExactMolWt(m))


def fp_and_mass(args):
    smi, _ = args
    r = fp_vec(smi)
    return r


def main():
    t0 = time.time()
    t = pq.read_table(TRAIN, columns=["inchikey14", "normalized_smiles"])
    df = t.to_pandas().dropna().drop_duplicates("inchikey14")
    print(f"[train-pool] {len(df):,} unique training structures {time.time()-t0:.0f}s", flush=True)
    tasks = [(smi, BITS) for smi in df.normalized_smiles.tolist()]
    with Pool(8) as mp:
        res = mp.map(fp_and_mass, tasks, chunksize=500)
    ok = [i for i, r in enumerate(res) if r is not None]
    fp = np.array([res[i][0] for i in ok], np.uint8)
    mass = np.array([res[i][1] for i in ok], np.float64)
    keys = df.inchikey14.values[ok]
    print(f"[train-pool] built {len(mass):,} fpg : {fp.shape} | mass {mass.min():.1f}-{mass.max():.1f} | {time.time()-t0:.0f}s", flush=True)
    print("[train-pool] wrote train_fp.npy / train_mass.npy / train_keys.npy")


if __name__ == "__main__":
    main()