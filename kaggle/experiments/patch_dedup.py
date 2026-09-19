import json, sys

SRC = '/tmp/nbs/analog-propagation-casmi-2026-baseline.ipynb'
OUT = '/tmp/graft/grafted.ipynb'
META = '/tmp/graft/kernel-metadata.json'

nb = json.load(open(SRC))
s = ''.join(nb['cells'][25]['source'])

i0 = s.find("order = np.argsort(-p)[:CFG.TOPN]")
i1 = s.find("smis  = [csmi[i] for i in order]")
assert i0 >= 0 and i1 > i0, (i0, i1)
suffix_len = len("smis  = [csmi[i] for i in order]")

NB = ("order = np.argsort(-p)\n"
      "            # ---- PHASE-1b graft: metric-exact final-topN dedup (grader-safe) ----\n"
      "            # Keep at most TOPN, dropping any candidate whose tautomer-canonical\n"
      "            # InChIKey14 (the exact key the grader matches) is already ranked higher.\n"
      "            smis  = []\n"
      "            seen  = set()\n"
      "            for _i in order:\n"
      "                if len(smis) >= CFG.TOPN:\n"
      "                    break\n"
      "                _smi = csmi[_i]\n"
      "                if HAVE_RDKIT and _smi:\n"
      "                    try:\n"
      "                        _m = Chem.MolFromSmiles(_smi)\n"
      "                        _k = None if _m is None else Chem.MolToInchiKey(_m)[:14]\n"
      "                    except Exception:\n"
      "                        _k = None\n"
      "                    if _k is not None:\n"
      "                        if _k in seen:\n"
      "                            continue\n"
      "                        seen.add(_k)\n"
      "                smis.append(_smi)")

s2 = s[:i0] + NB + s[i1 + suffix_len:]
assert "PHASE-1b graft" in s2
assert "order = np.argsort(-p)" in s2
assert s2.count("smis  = []") == 1
# downstream fallback must still exist
assert "if not smis: smis = ['CCO']" in s2 or "if not smis:" in s2

nb['cells'][25]['source'] = [s2]

import os
os.makedirs('/tmp/graft', exist_ok=True)
json.dump(nb, open(OUT, 'w'))

md = {
  "id": "aayanshabbir/casmi26-grafted-dedup",
  "title": "CASMI26 grafted dedup",
  "code_file": "grafted.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_internet": "false",
  "competition_sources": ["enveda-casmi26-molecule-id-mass-spectra"],
  "dataset_sources": [
    "prvsiyan/casmi26-fp-models-v2",
    "prvsiyan/casmi26-ranker-features",
    "prvsiyan/chebi-lipidmaps-casmi26",
    "prvsiyan/coconut-casmi26-candidates",
    "prvsiyan/pubchem-npformula-massindex",
    "prvsiyan/rdkit-wheel-offline"]
}
json.dump(md, open(META, 'w'), indent=1)

# re-open + verify persistence
nb2 = json.load(open(OUT))
s3 = ''.join(nb2['cells'][25]['source'])
print("persisted marker:", "PHASE-1b graft" in s3)
print("cells:", len(nb2['cells']))
print("cell25 lines:", len(s3.splitlines()))
print("META:", OUT, "exists:", os.path.exists(META))
print("OK")