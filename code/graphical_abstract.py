"""Graphical abstract (Elsevier: landscape, readable at 13 x 5 cm). All numbers read from the results files."""
import json, re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import params as P, model as M
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
mac = {}
for f in ("../tables/macros.tex", "../tables/macros2.tex"):
    for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(f).read()): mac[k] = v
R = json.load(open("../results/results.json")); d0 = R["det_superellipse"]["GDINO-B (semi)"]
sh = np.array(d0["shapes"]); r1c = np.array(d0["r_f1"]); r = np.linspace(0, 0.98, 197)
Rg = np.tile(r1c, (r.size, 1)); Rg[:, 0] = r; o = M.evaluate(Rg, sh, dict(P.BASE), True)

fig = plt.figure(figsize=(10.24, 3.94))          # 26 x 10 cm, 2:1 over Elsevier 13 x 5 cm minimum
ax0 = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax0.axis("off"); ax0.set_xlim(0, 100); ax0.set_ylim(0, 40)
GREEN, GOLD = "#5b7f4a", "#8a6d00"
def box(x, y, w, h, fc, ec="#444444"):
    ax0.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=1.0))
# ---- left: pathway
steps = [("Mixed paper-line\nfeed (1 t)", "#e8f1e4"), ("AI detection\nthreshold $r_c$", "#e3edf7"), ("Robotic picking", "#e8f1e4"), ("Bales and\npaper product", "#fdebd3")]
for k, (t, c) in enumerate(steps):
    y = 30.5 - k * 8.6
    box(1.5, y, 22, 6.4, c); ax0.text(12.5, y + 3.2, t, ha="center", va="center", fontsize=12, linespacing=1.05)
    if k < 3:
        ax0.add_patch(FancyArrowPatch((12.5, y - 0.4), (12.5, y - 1.8), arrowstyle="simple,head_width=0.9,head_length=0.8", mutation_scale=9, color=GREEN))
ax0.add_patch(FancyArrowPatch((24.5, 7), (24.5, 33), connectionstyle="arc3,rad=0.45", arrowstyle="-|>", mutation_scale=14, lw=1.6, ls=(0, (4, 2)), color=GOLD))
ax0.text(31.5, 20, "grade-aware\nthreshold\ntuning", ha="center", va="center", fontsize=10.5, style="italic", color="#6b5500", linespacing=1.05)
# ---- middle: grade cliff
ax = fig.add_axes([0.435, 0.20, 0.235, 0.62])
nsv = o["nsv"]; g = o["grade_line"]
ax.fill_between(r, 25, 50, where=g == 2, color="#d9f0d3", step="mid", lw=0)
ax.fill_between(r, 25, 50, where=g < 2, color="#fee0b6", step="mid", lw=0)
ax.plot(r, nsv, color="k", lw=2.2)
ax.set_ylim(29, 47.5); ax.set_xlim(0, 0.98)
ax.set_xlabel("Cardboard recall (AI threshold)", fontsize=11); ax.set_ylabel("Value (USD t$^{-1}$)", fontsize=11)
ax.tick_params(labelsize=10); [ax.spines[s].set_visible(False) for s in ("top", "right")]
ax.text(0.03, 46.8, "mixed-paper\ngrade", fontsize=10, va="top"); ax.text(0.97, 33.5, "sorted residential\npaper grade", fontsize=10, ha="right")
ax.annotate(f"grade cliff\n+{float(mac['CliffJump']):.0f} USD t$^{{-1}}$", xy=(float(mac["CrossRecall"]), 38), xytext=(0.04, 39.5),
            fontsize=11, weight="bold", arrowprops=dict(arrowstyle="->", lw=1.3))
ax.set_title("Crossing the grade cliff", fontsize=13, weight="bold", pad=6)
# ---- right: outcomes
box(71.5, 3, 27, 34, "#fff8e1", ec=GOLD)
ax0.text(85, 34.2, "Grade-aware vs. F1 thresholds", ha="center", va="center", fontsize=10.5, weight="bold")
rows = [("Robot picks per tonne", f"{mac['RFpicks']} vs {mac['PicksFone']}"),
        ("Misdirected", f"{mac['RFmis']} vs {mac['Fmis']} kg t$^{{-1}}$"),
        ("Film bale purity", f"{mac['RFfilm']}% (A) vs {mac['Ffilm']}% (C)"),
        ("Paper grade kept*", f"{mac['MCrfSRPN']}% vs {100-int(mac['MCfail'])}%")]
for k, (a, b) in enumerate(rows):
    y = 28.3 - k * 6.4
    ax0.text(73.3, y, a, fontsize=10.5, va="center", color="#444444")
    ax0.text(73.3, y - 2.7, b, fontsize=12, va="center", weight="bold")
ax0.text(73.3, 4.6, "*share of 2,000 uncertainty draws", fontsize=8.5, style="italic", color="#555555")
for ext in ("pdf", "svg"):
    fig.savefig(f"../submission/graphical_abstract.{ext}")
fig.savefig("../submission/graphical_abstract.png", dpi=300)
fig.savefig("../submission/graphical_abstract.tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
print("GA ok")
