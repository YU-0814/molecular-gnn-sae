"""Section 4.3: Cohen's d of drug-class-aligned SAE features (f690/f819/f675) vs the best GEM neuron (sign sweep), plus top-3 molecules per feature."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from common import DATA, RESULTS, JumpReLUSAE, cohens_d, load_activations

HEROES = [
    # (약물 클래스, SAE feature, 이름 패턴)
    ("Corticosteroid", 690,
     lambda n: any(p in n for p in ["predn", "cortis", "asone", "olone", "methasone", "cortone", "fluticasone"])),
    ("Inhalational anesthetic", 819, lambda n: "flurane" in n),
    ("Anthracycline", 675, lambda n: "rubicin" in n),
]
N_SHOW = 3


def main():
    X, gids, _ = load_activations("bbbp")
    meta = pd.read_csv(DATA / "activations/bbbp/molecule_meta.csv").set_index("graph_id").loc[gids]
    names = meta["name"].fillna("").str.lower().values
    F = JumpReLUSAE().encode(torch.from_numpy(X)).numpy()
    print(f"BBBP N={len(X)}  alive features={(F.sum(0) > 0).sum()}/1024  L0={(F > 0).sum(1).mean():.1f}")

    out = []
    for label, fi, is_target in HEROES:
        mask = np.array([is_target(n) for n in names])
        neuron_d = [cohens_d(X[:, n], mask) for n in range(32)]
        bn = int(np.argmax(np.abs(neuron_d)))  # sign sweep: 절댓값 최대
        vals = F[:, fi]
        top = np.where(mask & (vals > 0))[0]
        top = top[np.argsort(vals[top])[::-1]][:N_SHOW]
        rec = dict(label=label, feature=fi, n_target=int(mask.sum()), n_active=int((vals > 0).sum()),
                   sae_cohens_d=cohens_d(vals, mask), best_neuron=bn, best_neuron_cohens_d=neuron_d[bn],
                   top_molecules=[dict(name=meta["name"].iloc[i], smiles=meta["canonical_smiles"].iloc[i],
                                       activation=float(vals[i])) for i in top])
        out.append(rec)
        print(f"{label:<24} f{fi}: SAE d={rec['sae_cohens_d']:6.2f} | GEM neuron {bn:2d} d={neuron_d[bn]:5.2f} "
              f"| n_target={rec['n_target']}  top={[m['name'] for m in rec['top_molecules']]}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "hero_features.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved results/hero_features.json")


if __name__ == "__main__":
    main()
