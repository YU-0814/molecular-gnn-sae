"""Table 2: ROC-AUC of a frozen-feature MLP head on raw GEM (32d) / SAE sparse (1024d) / SAE reconstruction (32d). Scaffold split, 100 epochs, 5 seeds."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

from common import DATASETS, RESULTS, JumpReLUSAE, load_activations, load_labels


class Head(nn.Module):
    def __init__(self, in_dim=32, hidden=128, n_tasks=1, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_tasks),
        )

    def forward(self, x):
        return self.mlp(self.norm(x))


def scaffold_split(smiles, frac_train=0.8, frac_val=0.1):
    """Bemis-Murcko scaffold 기준 분할 (큰 scaffold 그룹부터 train에 채움)."""
    groups = {}
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=True)
        groups.setdefault(sc, []).append(i)
    sets = sorted((sorted(v) for v in groups.values()), key=lambda s: (len(s), s[0]), reverse=True)
    n = len(smiles)
    tr, va, te = [], [], []
    for s in sets:
        if len(tr) + len(s) > frac_train * n:
            (te if len(tr) + len(va) + len(s) > (frac_train + frac_val) * n else va).extend(s)
        else:
            tr.extend(s)
    return np.array(tr), np.array(va), np.array(te)


def multitask_auc(y, s):
    return np.array([roc_auc_score(y[:, k], s[:, k]) if len(np.unique(y[:, k])) > 1 else np.nan
                     for k in range(y.shape[1])])


def train_head(X, Y, tr, va, te, seed, device, in_dim, epochs=100, lr=1e-3, batch=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt, Yt = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
    n_tasks = Y.shape[1]
    head = Head(n_tasks=n_tasks).to(device)
    proj = nn.Linear(in_dim, 32).to(device) if in_dim != 32 else nn.Identity()
    model = nn.Sequential(proj, head)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    best_val, best_state = -np.inf, None
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(len(tr), device=device)
        for i in range(0, len(idx), batch):
            b = tr[idx[i:i + batch].cpu().numpy()]
            loss = crit(model(Xt[b]), Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.nanmean(multitask_auc(Y[va], model(Xt[va]).cpu().numpy())))
        if v > best_val:
            best_val, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        aucs = multitask_auc(Y[te], model(Xt[te]).cpu().numpy())
    return dict(val_auc=best_val, test_auc=aucs.tolist(), test_auc_mean=float(np.nanmean(aucs)))


def run_task(task, n_seeds, device, sae):
    X, _, smiles = load_activations(task)
    Y = load_labels(task, smiles)
    keep = ~np.isnan(Y).any(axis=1)
    X, Y, smiles = X[keep], Y[keep], [s for s, k in zip(smiles, keep, strict=True) if k]
    tr, va, te = scaffold_split(smiles)
    F = sae.encode(torch.from_numpy(X).to(device))
    reps = {"raw": (X, 32), "sparse": (F.cpu().numpy(), F.shape[1]), "recon": (sae.decode(F).cpu().numpy(), 32)}
    print(f"[{task}] N={len(X)} split={len(tr)}/{len(va)}/{len(te)}  L0={(reps['sparse'][0] > 0).sum(1).mean():.1f}", flush=True)
    out = {"task_cols": DATASETS[task]["task_cols"], "n_seeds": n_seeds}
    for name, (R, d) in reps.items():
        runs = [train_head(R, Y, tr, va, te, s, device, d) for s in range(n_seeds)]
        aucs = [r["test_auc_mean"] for r in runs]
        out[name] = dict(auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)), seeds=runs)
        print(f"  {name:<7} AUC = {np.mean(aucs):.4f} ({np.std(aucs):.4f})", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(DATASETS))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    device = torch.device(a.device)
    sae = JumpReLUSAE(device=a.device)
    t0 = time.time()
    results = {t: run_task(t, a.seeds, device, sae) for t in a.tasks}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "downstream_auc.json").write_text(json.dumps(results, indent=2))
    print("\n표 2 (ROC-AUC, std)         " + "  ".join(f"{t:>16}" for t in a.tasks))
    for rep, label in [("raw", "원본 32"), ("sparse", "희소 32→1024"), ("recon", "복원 32→1024→32")]:
        row = "  ".join(f"{results[t][rep]['auc_mean']:.4f} ({results[t][rep]['auc_std']:.4f})" for t in a.tasks)
        print(f"{label:<22} {row}")
    print(f"saved results/downstream_auc.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
