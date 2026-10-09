import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, PathPatch
from matplotlib.path import Path
import params as P, model as M
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8, "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False})
R2 = json.load(open("../results/results_revision.json")); FD = "../figures_intermediate/"
C = dict(paper="#9e9e9e", cb="#8c510a", sp="#01665e", rp="#5e3c99", m="#b2182b")
def save(fig, name):
    fig.savefig(FD + name + ".svg", bbox_inches="tight"); fig.savefig(FD + name + ".pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 1 is produced by fig1_framework.py
exec(open("fig1_framework.py").read())

# ---------------- Material-flow (Sankey) figure is produced by fig9_sankey.py
exec(open("fig9_sankey.py").read())

# ---------------- Fig: Pareto front value vs recovered mass
fig, ax = plt.subplots(figsize=(4.8, 3.4)); cl = R2["cloud"]
ax.scatter(cl["rec"], cl["nsv"], s=2, color="#cccccc", label="Grid operating points", rasterized=True)
fr = np.array(R2["pareto"]); ax.plot(fr[:, 2], fr[:, 1], color="k", lw=1.3, label="Pareto front")
mk = {"No QC": "s", "F1-optimal": "o", "High-precision (p>=0.90)": "^", "Uniform recall 0.80": "v", "Global threshold (tuned)": "D",
      "Value-optimal": "*", "Value-optimal, recovery floor": "P"}
for n, d in R2["policies"].items(): ax.scatter(d["recovered_total"], d["nsv"], marker=mk[n], s=45, zorder=4, label=n)
ax.set_ylim(min(28, min(cl["nsv"])), 47); ax.set_xlabel("Target material recovered to its own bin (kg t$^{-1}$ feed)"); ax.set_ylabel("Net sales value (USD t$^{-1}$ feed)")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0)); save(fig, "fig10_pareto")

# ---------------- Fig: scenarios
sc = R2["scenarios"]; ws = ["Low (4%)", "Medium (8%)", "High (14%)"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, spec in zip(axs, ("Standard", "Strict")):
    x = np.arange(3); wdt = 0.12
    for k, (pn, col) in enumerate((("F1-optimal", "#2c7bb6"), ("Global threshold (tuned)", "#abd9e9"), ("Value-optimal", "#d7191c"), ("Value-optimal, recovery floor", "#fdae61"))):
        for q, (pl, hatch) in enumerate((("2026-06", ""), ("2025-06", "///"))):
            v = [sc[f"{w}|{pl}|{spec}"][pn]["nsv"] - sc[f"{w}|{pl}|{spec}"]["No QC"]["nsv"] for w in ws]
            ax.bar(x + (k * 2 + q - 3.5) * wdt, v, width=wdt, color=col, hatch=hatch, edgecolor="white", lw=0.3,
                   label=(pn if q == 0 else None))
    ax.axhline(0, color="k", lw=0.7); ax.set_xticks(x); ax.set_xticklabels(ws); ax.set_title(f"{spec} bale specification", fontsize=8)
    ax.set_xlabel("Target (contaminant) mass share of feed")
axs[0].set_ylabel("Value of robotic QC vs. no QC (USD t$^{-1}$)"); axs[1].legend(fontsize=8, loc="upper left")
axs[0].text(0.02, 0.97, "solid: 2026 prices; hatched: 2025 prices", transform=axs[0].transAxes, fontsize=8, va="top")
fig.tight_layout(); save(fig, "fig11_scenarios")

# ---------------- Fig: Monte Carlo + convergence
mc = R2["mc"]
fig, axs = plt.subplots(1, 3, figsize=(8.2, 3.0))
axs[0].hist(mc["g_f1"], bins=40, color="#2c7bb6", alpha=0.65, label="F1-optimal")
axs[0].hist(mc["g_rf"], bins=40, color="#fdae61", alpha=0.65, label="Value-opt., recovery floor")
axs[0].hist(mc["g_opt"], bins=40, color="#d7191c", alpha=0.45, label="Value-optimal")
axs[0].set_xlabel("Value of QC (USD t$^{-1}$)"); axs[0].set_ylabel("Draws"); axs[0].legend(fontsize=8)
axs[1].hist(mc["tau"], bins=20, color="#636363"); axs[1].set_xlabel("Kendall $\\tau$ (mAP vs value rank)")
g = np.array(mc["g_rf"]) - np.array(mc["g_f1"]); fail = 1 - np.array(mc["srpn_f1"]); nn = np.arange(1, len(g) + 1)
axs[2].plot(nn, np.cumsum(g) / nn, color="k", label="mean gain over F1 (USD t$^{-1}$)")
ax2 = axs[2].twinx(); ax2.plot(nn, np.cumsum(fail) / nn, color="#d7191c", lw=1, label="share of draws F1 loses SRPN")
ax2.set_ylim(0, 1); ax2.tick_params(labelsize=8); ax2.spines["right"].set_visible(True)
axs[2].set_xlabel("Number of draws"); axs[2].set_ylabel("Running mean (USD t$^{-1}$)")
axs[2].set_ylim(0, 12); axs[2].legend(fontsize=8, loc="lower left"); ax2.legend(fontsize=8, loc="lower right")
for a, t in zip(axs, "abc"): a.text(-0.2, 1.03, f"({t})", transform=a.transAxes, weight="bold")
fig.tight_layout(); save(fig, "fig8_monte_carlo")
print("ok")
