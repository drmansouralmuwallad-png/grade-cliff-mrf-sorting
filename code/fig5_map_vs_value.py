"""Figure: mAP versus value for twelve detectors, dumbbell chart with a broken value axis."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import params as P
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
R = json.load(open("../results/results.json")); det = R["det_superellipse"]
names = sorted(P.DETECTORS, key=lambda n: P.DETECTORS[n][1])      # highest mAP at top
mAP = [P.DETECTORS[n][1] for n in names]
f1 = np.array([det[n]["nsv_f1"] for n in names]); vo = np.array([det[n]["nsv_opt"] for n in names])
none = det["GDINO-B (semi)"]["none"]; y = np.arange(len(names))
L0, L1, R0, R1 = 29.6, 33.0, 43.9, 46.1
fig, (axl, axr) = plt.subplots(1, 2, sharey=True, figsize=(7.2, 4.8), gridspec_kw=dict(width_ratios=[1, 3.2], wspace=0.06))
BLUE, RED, GREY = "#2c7bb6", "#d7191c", "#8f8f8f"
for ax, lo, hi in ((axl, L0, L1), (axr, R0, R1)):
    for i in range(len(names)):
        a, b = sorted((f1[i], vo[i]))
        ax.plot([max(a, lo - 1), min(b, hi + 1)], [y[i], y[i]], color=GREY, lw=1.6, zorder=1, clip_on=True)
    ax.scatter(f1, y, s=48, facecolor="white", edgecolor=BLUE, lw=1.6, zorder=3, clip_on=True)
    ax.scatter(vo, y, s=95, marker="*", color=RED, zorder=4, clip_on=True)
    ax.set_xlim(lo, hi); ax.grid(axis="x", color="#e3e3e3", lw=0.6); ax.set_axisbelow(True)
    for i in range(len(names)):
        if i % 2 == 0: ax.axhspan(i - 0.5, i + 0.5, color="#f7f7f7", zorder=0)
axl.axvline(none, color="k", ls=":", lw=1.1)
axl.text(none - 0.12, len(names) - 0.55, f"{none:.1f}", fontsize=8, ha="right", va="top")
axl.set_xticks([30, 31, 32, 33]); axr.set_xticks([44, 44.5, 45, 45.5, 46])
for sp in ("top", "right"): axl.spines[sp].set_visible(False)
for sp in ("top", "left", "right"): axr.spines[sp].set_visible(False)
axr.tick_params(axis="y", length=0)
# break marks
d = 0.012
for ax, xs in ((axl, (1 - d * 3.2, 1 + d * 3.2)), (axr, (-d, d))):
    ax.plot(xs, (-d * 1.2, d * 1.2), transform=ax.transAxes, color="k", lw=0.9, clip_on=False)
# y labels: detector name and mAP in two aligned columns
axl.set_yticks(y); axl.set_yticklabels(names, fontsize=8.5); axl.tick_params(axis="y", length=0, pad=46)
for i, m in enumerate(mAP):
    axl.text(-0.05, y[i], f"{m:.1f}", transform=axl.get_yaxis_transform(), ha="right", va="center", fontsize=8.5)
axl.text(-0.05, len(names) - 0.35, "mAP (%)", transform=axl.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.5, weight="bold")
import matplotlib.transforms as mt
axl.text(0, len(names) - 0.35, "Detector", transform=mt.offset_copy(axl.get_yaxis_transform(), fig=fig, x=-50, units="points"), ha="right", va="bottom", fontsize=8.5, weight="bold")
# gain column at right
for i in range(len(names)):
    axr.text(1.02, y[i], f"+{vo[i]-f1[i]:.1f}" if vo[i] - f1[i] < 5 else f"+{vo[i]-f1[i]:.1f}", transform=axr.get_yaxis_transform(),
             ha="left", va="center", fontsize=8.5, color="#333333")
axr.text(1.02, len(names) - 0.35, "Gain", transform=axr.get_yaxis_transform(), ha="left", va="bottom", fontsize=8.5, weight="bold")
axl.set_ylim(-0.5, len(names) - 0.5)
fig.supxlabel("Net sales value (USD t$^{-1}$ feed)", fontsize=9, y=0.02)
handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, mew=1.6, ms=7, label="F1-optimal thresholds"),
           Line2D([], [], marker="*", ls="", color=RED, ms=11, label="Value-optimal thresholds"),
           Line2D([], [], ls=":", color="k", label="No robotic QC")]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.0), frameon=False, fontsize=8.5)
fig.subplots_adjust(top=0.9)
for ext in ("svg", "pdf"):
    fig.savefig(f"../figures_intermediate/fig5_map_vs_value.{ext}", bbox_inches="tight")
print("fig5 ok")
