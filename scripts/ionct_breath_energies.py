"""Per-frame breathing energy of complete-model runs over the training cache (for figures/ionct_m25/breathing.py).

    ionct_breath_energies.py --ct-sets DIR --walk-eda DIR --params ions.json --cache cache.pt --runs RUN_DIR... --out breathing_energies.json

Writes {run name: [{workflow, parent, n, pol_ref (kJ/mol), e_breath (kJ/mol), e_pol (kJ/mol)}]}; frames are keyed the
same way as predictions.csv (workflow, parent, n, pol_ref)."""
import argparse
import json
from pathlib import Path

import torch

from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.model import collate
from rsfff.ff.ionct.params import IonTable
from rsfff.train.train_ionct import KJ, build_model, load_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-sets", required=True)
    ap.add_argument("--walk-eda", default=None)
    ap.add_argument("--params", required=True)
    ap.add_argument("--cache", required=True, help="the lean+frontier density cache of these frames (train_ionct --cache-dir)")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="breathing_energies.json")
    ap.add_argument("--ct-max-pair", type=float, default=3.6)
    ap.add_argument("--max-ho-ratio", type=float, default=1.4)
    a = ap.parse_args()
    tab, chan = IonTable.load(a.params), CT.ChannelTable.load(a.params)
    frames = load_frames(a.ct_sets, a.walk_eda, a.ct_max_pair, a.max_ho_ratio)
    caches = torch.load(a.cache, weights_only=False)
    assert len(caches) == len(frames), "cache / frame mismatch"
    for c, (sym, X, lab, meta) in zip(caches, frames):
        c.labels, c.meta = dict(lab), dict(meta)
    out = {}
    for run in a.runs:
        ck = torch.load(Path(run) / "model.pt", weights_only=False)
        m = build_model(ck["args"], tab, chan, ck["n_feat"], ck["state"]["feat_scale"], ck.get("feature_names"))
        m.load_state_dict(ck["state"])
        m.eval()
        rows = []
        with torch.no_grad():
            for k in range(0, len(caches), 64):
                chunk = caches[k:k + 64]
                bt = collate(chunk)
                o = m(bt)
                eb = torch.zeros(bt["B"], dtype=torch.float64).index_add(0, bt["frame"], o["e_breath_atom"])
                for q, f in enumerate(chunk):
                    rows.append({"workflow": f.meta["workflow"], "parent": f.meta["parent"], "n": f.meta["n"],
                                 "pol_ref": round(f.labels["pol"] * KJ, 6), "e_breath": eb[q].item() * KJ,
                                 "e_pol": o["e_pol"][q].item() * KJ})
        out[Path(run).name] = rows
        print(f"{Path(run).name}: {sum(abs(r['e_breath']) > 1 for r in rows)} frames with |E_breath| > 1 kJ/mol, "
              f"min {min(r['e_breath'] for r in rows):.1f}", flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
