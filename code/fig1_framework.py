"""Figure 1: resource-recovery framework (physical pathway, model quantities, optimisation loop)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.48, 4.9))
ax.set_xlim(0, 100); ax.set_ylim(-6, 64); ax.axis("off")
PHYS, INFO, OUT, OPT = "#e8f1e4", "#e3edf7", "#fdebd3", "#fff4c2"
GREEN, GOLD = "#5b7f4a", "#8a6d00"
W, H = 20.5, 23
cols = [1, 26, 51]; rows = [31, 3]
stages = [
    ("1", "Mixed feed", "1 t of paper-line\nfeed: paper with\ncardboard, film,\nrigid plastic, metal", PHYS),
    ("2", "AI detection", "precision–recall\ncurve per class;\noperating recall $r_c$\n= AI threshold", INFO),
    ("3", "Robotic\nseparation", "correct picks,\nmissed objects,\nfalse picks;\nitem balance", PHYS),
    ("4", "Material\nstreams", "kg t$^{-1}$ to paper\nproduct and to OCC,\nfilm, rigid and\nmetal bins", PHYS),
    ("5", "Purity and\ncontamination", "stream purity;\nresidual and\nmisdirected mass\n(kg t$^{-1}$)", PHYS),
    ("6", "Bale grade\nand value", "specification grade\n(ReMA, APR);\nrecovered kg t$^{-1}$;\nvalue USD t$^{-1}$", OUT),
]
pos = [(cols[0], rows[0]), (cols[1], rows[0]), (cols[2], rows[0]), (cols[0], rows[1]), (cols[1], rows[1]), (cols[2], rows[1])]
for (x, y), (num, title, body, col) in zip(pos, stages):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.3,rounding_size=1.2", fc=col, ec="#444444", lw=0.9))
    ty = y + H - 5.0
    ax.add_patch(plt.Circle((x + 2.3, ty), 1.6, color="#333333", zorder=3))
    ax.text(x + 2.3, ty, num, color="white", ha="center", va="center", fontsize=8.5, weight="bold", zorder=4)
    ax.text(x + 4.4 + (W - 4.4) / 2, ty, title, ha="center", va="center", fontsize=8.5, weight="bold", linespacing=1.1)
    ly = y + H - 9.8
    ax.plot([x + 1.2, x + W - 1.2], [ly, ly], color="#aaaaaa", lw=0.6)
    ax.text(x + W / 2, (y + ly) / 2 + 0.3, body, ha="center", va="center", fontsize=8, linespacing=1.28)
def flow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="simple,head_width=1.2,head_length=1.0", mutation_scale=8, color=GREEN, zorder=5))
for (x0, y0), (x1, _) in (((cols[0], rows[0]), (cols[1], 0)), ((cols[1], rows[0]), (cols[2], 0)),
                          ((cols[0], rows[1]), (cols[1], 0)), ((cols[1], rows[1]), (cols[2], 0))):
    flow((x0 + W + 0.4, y0 + H / 2), (x1 - 0.4, y0 + H / 2))
# connector 3 -> 4 (elbow)
xm = cols[2] + W / 2
ax.plot([xm, xm], [rows[0] - 0.4, 29], color=GREEN, lw=3.2, solid_capstyle="butt")
ax.plot([xm + 0.8, cols[0] + W / 2 - 0.8], [29, 29], color=GREEN, lw=3.2)
ax.plot([xm, xm + 0.8], [29, 29], color=GREEN, lw=3.2)
flow((cols[0] + W / 2, 29.8), (cols[0] + W / 2, rows[1] + H + 0.4))
# optimisation box (right)
ox, oy, ow, oh = 80, 9, 19.5, 42
ax.add_patch(FancyBboxPatch((ox, oy), ow, oh, boxstyle="round,pad=0.3,rounding_size=1.2", fc=OPT, ec=GOLD, lw=1.0))
ax.text(ox + ow / 2, oy + oh - 5, "Operating-point\noptimisation", ha="center", va="center", fontsize=8.5, weight="bold", linespacing=1.1)
ax.plot([ox + 1.2, ox + ow - 1.2], [oy + oh - 9.6, oy + oh - 9.6], color="#c9b25a", lw=0.6)
ax.text(ox + ow / 2, oy + 17.5, "choose $r_c$ for\neach class to\nmaximise value\n\nsubject to:\nrobot pick budget\nand a recovery\nfloor (kg t$^{-1}$)",
        ha="center", va="center", fontsize=8, linespacing=1.28)
kw = dict(arrowstyle="-|>", mutation_scale=11, lw=1.3, color=GOLD, zorder=5)
ax.add_patch(FancyArrowPatch((cols[2] + W + 0.4, rows[1] + H / 2), (ox - 0.4, rows[1] + H / 2), linestyle=(0, (4, 2)), **kw))
ax.text((cols[2] + W + ox) / 2, rows[1] + H / 2 - 2.0, "grade,\nrecovery,\nvalue", ha="center", va="top", fontsize=8, style="italic", color="#6b5500", linespacing=1.05)
# feedback to AI thresholds, routed above the top row
yt = 59
ax.plot([ox + ow / 2, ox + ow / 2], [oy + oh + 0.4, yt], color=GOLD, lw=1.3, ls=(0, (4, 2)))
ax.plot([ox + ow / 2, cols[1] + W / 2], [yt, yt], color=GOLD, lw=1.3, ls=(0, (4, 2)))
ax.add_patch(FancyArrowPatch((cols[1] + W / 2, yt), (cols[1] + W / 2, rows[0] + H + 0.5), linestyle=(0, (4, 2)), **kw))
ax.text(64, yt + 1.2, "updated AI operating thresholds $r_c$", ha="center", va="bottom", fontsize=8, style="italic", color="#6b5500")
# legend (bottom)
ly = -3.8
flow((1, ly), (6.5, ly)); ax.text(7.5, ly, "material flow", va="center", fontsize=8)
ax.plot([24, 29.5], [ly, ly], color=GOLD, lw=1.3, ls=(0, (4, 2))); ax.text(30.5, ly, "decision feedback", va="center", fontsize=8)
for x0, col, lab in ((51, PHYS, "physical process"), (69, INFO, "AI layer"), (82, OUT, "market outcome")):
    ax.add_patch(FancyBboxPatch((x0, ly - 1.4), 2.8, 2.8, boxstyle="round,pad=0.1", fc=col, ec="#444444", lw=0.6))
    ax.text(x0 + 3.9, ly, lab, va="center", fontsize=8)
for ext in ("svg", "pdf"):
    fig.savefig(f"../figures_intermediate/fig1_framework.{ext}", bbox_inches="tight")
print("fig1 ok")
