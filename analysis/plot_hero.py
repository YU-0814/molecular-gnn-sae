"""Figures 3 and 4 from results/hero_features.json: activation scatter (SAE feature vs GEM neuron) and top molecules."""
from __future__ import annotations

import io
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from common import DATA, RESULTS, JumpReLUSAE, load_activations
from hero_features import HEROES

COLORS = {"Corticosteroid": "#c0392b", "Inhalational anesthetic": "#1a5276", "Anthracycline": "#1e6b3e"}
plt.rcParams.update({"font.family": "serif", "font.size": 18, "axes.linewidth": 1.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def draw_mol(smi, px=1200, pad=4):
    """Render a molecule, crop the whitespace and center it on a square canvas."""
    mol = Chem.MolFromSmiles(smi)
    drawer = rdMolDraw2D.MolDraw2DCairo(px, px)
    o = drawer.drawOptions()
    o.padding, o.bondLineWidth, o.minFontSize, o.maxFontSize = 0.08, 14.0, 64, 90
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    arr = np.array(Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB"))
    ink = (arr < 248).any(axis=2)
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    arr = arr[max(0, rows[0] - pad):rows[-1] + 1 + pad, max(0, cols[0] - pad):cols[-1] + 1 + pad]
    h, w = arr.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = arr
    return canvas


def wrap_name(name, width=14):
    parts = textwrap.wrap(name.strip(), width=width, break_on_hyphens=True)
    return "\n".join(parts[:1] + [" ".join(parts[1:])] if len(parts) > 1 else parts)


def scatter(ax, vals, mask, xlabel, d, color, rng):
    ax.scatter(vals[~mask], rng.uniform(0.02, 0.98, (~mask).sum()), s=16,
               facecolors="#cccccc", edgecolors="#555555", linewidths=0.5, alpha=0.7, zorder=1)
    ax.scatter(vals[mask], rng.uniform(0.02, 0.98, mask.sum()), s=64,
               facecolors=color, edgecolors="#222222", linewidths=0.6, alpha=0.92, zorder=2)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_yticks([])
    ax.set_ylabel("BBBP dataset", fontsize=17)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_title(f"Cohen's d = {d:.2f}", fontsize=18.5, fontweight="bold", color=color, pad=6)


def row_header(fig, spec, text, color):
    fig.canvas.draw()
    p = spec.get_position(fig)
    fig.text((p.x0 + p.x1) / 2, p.y1 + 0.014, text, ha="center", va="bottom", fontsize=23, fontweight="bold", color=color)


def main():
    heroes = json.loads((RESULTS / "hero_features.json").read_text())
    X, gids, _ = load_activations("bbbp")
    names = pd.read_csv(DATA / "activations/bbbp/molecule_meta.csv").set_index("graph_id").loc[gids]["name"].fillna("").str.lower().values
    F = JumpReLUSAE().encode(torch.from_numpy(X)).numpy()
    is_target = {label: fn for label, _, fn in HEROES}
    rng = np.random.default_rng(42)

    # Figure 3: activation scatter, SAE feature vs. best GEM neuron
    fig = plt.figure(figsize=(8.0, 12.4))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.30, left=0.10, right=0.98, top=0.93, bottom=0.06)
    for r, h in enumerate(heroes):
        mask = np.array([is_target[h["label"]](n) for n in names])
        color = COLORS[h["label"]]
        scatter(fig.add_subplot(gs[r, 0]), F[:, h["feature"]], mask, f"SAE Feature {h['feature']}", h["sae_cohens_d"], color, rng)
        scatter(fig.add_subplot(gs[r, 1]), X[:, h["best_neuron"]], mask, f"GEM Neuron {h['best_neuron']}", h["best_neuron_cohens_d"], color, rng)
    fig.canvas.draw()
    for col, text in enumerate(["SAE Feature", "GEM Neuron (best)"]):
        p = gs[0, col].get_position(fig)
        fig.text((p.x0 + p.x1) / 2, 0.985, text, ha="center", va="top", fontsize=22, fontweight="bold")
    fig.savefig(RESULTS / "fig3_sae_feature_vs_gem_neuron.png", dpi=200, bbox_inches="tight")

    # Figure 4: top-activating molecules per feature
    fig = plt.figure(figsize=(10.5, 12.4))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.78, left=0.03, right=0.99, top=0.92, bottom=0.04)
    for r, h in enumerate(heroes):
        mols = h["top_molecules"]
        inner = gridspec.GridSpecFromSubplotSpec(1, len(mols), subplot_spec=gs[r, 0], wspace=0.06, hspace=0.04)
        for c, m in enumerate(mols):
            ax = fig.add_subplot(inner[0, c])
            ax.imshow(draw_mol(m["smiles"]))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_xlabel(wrap_name(m["name"]), fontsize=20, labelpad=6)
        row_header(fig, gs[r, 0], f"Feature {h['feature']}: {h['label']}", COLORS[h["label"]])
    fig.savefig(RESULTS / "fig4_feature_top_molecules.png", dpi=200, bbox_inches="tight")
    print("saved results/fig3_sae_feature_vs_gem_neuron.png, results/fig4_feature_top_molecules.png")


if __name__ == "__main__":
    main()
