"""Shared: paths, JumpReLU SAE loader (no SAELens needed), activation/label loading, Cohen's d. Everything runs from data/ (5.8MB)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from safetensors import safe_open

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DATASETS = {
    "bbbp":    dict(csv="BBBP.csv",    smi_col="smiles", task_cols=["p_np"]),
    "clintox": dict(csv="clintox.csv", smi_col="smiles", task_cols=["FDA_APPROVED", "CT_TOX"]),
    "bace":    dict(csv="bace.csv",    smi_col="mol",    task_cols=["Class"]),
}


class JumpReLUSAE:
    """Minimal JumpReLU SAE that reads SAELens inference-format weights (no SAELens dependency)."""

    def __init__(self, sae_dir: Path = DATA / "sae", device: str = "cpu"):
        with safe_open(sae_dir / "sae_weights.safetensors", framework="pt") as f:
            self.W_enc = f.get_tensor("W_enc").to(device)      # (32, 1024)
            self.b_enc = f.get_tensor("b_enc").to(device)      # (1024,)
            self.W_dec = f.get_tensor("W_dec").to(device)      # (1024, 32)
            self.b_dec = f.get_tensor("b_dec").to(device)      # (32,)
            self.thresh = f.get_tensor("threshold").to(device) # (1024,)
        cfg = json.loads((sae_dir / "cfg.json").read_text())
        self.apply_b_dec = cfg.get("apply_b_dec_to_input", True)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_b_dec:
            x = x - self.b_dec
        pre = x @ self.W_enc + self.b_enc
        return pre * (pre > self.thresh).float()

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec


def canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else smi


def load_activations(task: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return GEM layer-8 graph embeddings X[N, 32], graph_ids[N] and SMILES[N], in graph_id order."""
    d = np.load(DATA / "activations" / task / "graph_layer08.npz")
    X, gids = d["activations"].astype(np.float32), d["graph_ids"]
    gid_to_smi = {}
    with open(DATA / "activations" / task / "graphs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            gid_to_smi[r["graph_id"]] = r["smiles"]
    return X, gids, [gid_to_smi[int(g)] for g in gids]


def load_labels(task: str, smiles: list[str]) -> np.ndarray:
    """Join MoleculeNet labels on canonical SMILES; molecules without a label get NaN."""
    cfg = DATASETS[task]
    df = pd.read_csv(DATA / "moleculenet" / cfg["csv"])
    label_map = {canon(row[cfg["smi_col"]]): row[cfg["task_cols"]].values.astype(np.float32)
                 for _, row in df.iterrows()}
    Y = np.full((len(smiles), len(cfg["task_cols"])), np.nan, dtype=np.float32)
    for i, s in enumerate(smiles):
        v = label_map.get(canon(s))
        if v is not None:
            Y[i] = v
    return Y


def cohens_d(vals: np.ndarray, mask: np.ndarray) -> float:
    """Cohen's d between target (mask) and non-target activations, pooled SD with ddof=1."""
    a, b = vals[mask], vals[~mask]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    pooled = math.sqrt(((len(a) - 1) * s1**2 + (len(b) - 1) * s2**2) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
