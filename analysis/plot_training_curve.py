"""Figure 2: SAE training curve (MSE, L0 vs epoch) from the W&B history CSV in data/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/training_curve_b1vcu50z.csv"
OUT = ROOT / "results/fig2_sae_training_curve.png"
N_EPOCH = 30

df = pd.read_csv(CSV).sort_values("_step")
epoch = df["_step"] / df["_step"].max() * N_EPOCH
final_mse = df["losses/mse_loss"].tail(50).mean()
final_l0 = df["metrics/l0"].tail(50).mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(epoch, df["losses/mse_loss"], color="#1f4e79", lw=1.6)
ax1.set_yscale("log")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Reconstruction MSE")
ax1.axhline(final_mse, ls=":", color="gray", lw=1)
ax1.annotate(f"MSE ≈ {final_mse:.1e}", xy=(N_EPOCH, final_mse), xytext=(14, final_mse * 30),
             color="#1f4e79", arrowprops=dict(arrowstyle="->", color="#1f4e79"))
ax1.set_title("(a)", loc="left", fontweight="bold")

ax2.plot(epoch, df["metrics/l0"], color="#1e6b3e", lw=2)
ax2.set_ylim(0, 200)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Mean Active Features ($L_0$)")
ax2.axhline(final_l0, ls=":", color="gray", lw=1)
ax2.annotate(f"$L_0$ = {final_l0:.1f}", xy=(N_EPOCH, final_l0), xytext=(18, 95),
             color="#1e6b3e", arrowprops=dict(arrowstyle="->", color="#1e6b3e"))
ax2.set_title("(b)", loc="left", fontweight="bold")

for ax in (ax1, ax2):
    ax.axvspan(18, N_EPOCH, color="#e8f5f0", alpha=0.4, lw=0)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"saved {OUT}  final MSE={final_mse:.2e}  final L0={final_l0:.2f}")
