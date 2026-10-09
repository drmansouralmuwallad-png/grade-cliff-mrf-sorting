"""Material-flow figure: where 1 t of paper-line feed goes under four operating policies.
Contaminant flows are drawn to one common scale; the large paper-to-paper-product flow is truncated."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D
import params as P, model as M
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
R2 = json.load(open("../results/results_revision.json"))
PR = P.PRICES[P.BASE["price"]]

def flows(pname):
    sh = M.shapes_for("GDINO-B (semi)", "superellipse"); r = np.array(R2["policies"][pname]["r"])
    o = M.evaluate(r, sh, dict(P.BASE), True)
    n = o["n"]; m = P.BASE["m"]; F = np.zeros((5, 5))
    prec = o["prec"]; TP = r * n; D = TP / prec; FP = D - TP; phi = P.BASE["phi"]; eta = P.BASE["eta"]
    W = np.tile(n, (4, 1)); np.fill_diagonal(W, 0); W = W / W.sum(1, keepdims=True)
    FPo = (1 - phi) * FP[:, None] * W; demand = TP + FPo.sum(0); s = np.minimum(1, n / np.maximum(demand, 1e-12))
    TPs = TP * s; FPos = FPo * s[None, :]
    for c in range(4):
        F[0, c + 1] = eta * phi * FP[c] * P.BASE["m_paper"]
        F[c + 1, c + 1] += eta * TPs[c] * m[c]
        for j in range(4): F[j + 1, c + 1] += eta * FPos[c, j] * m[j]
    F[0, 0] = 1000 * (1 - P.BASE["w"]) - F[0, 1:].sum()
    for j in range(4): F[j + 1, 0] = o["resid"][j]
    return F, o

SRC = ["Paper", "Cardboard", "Soft plastic", "Rigid plastic", "Metal"]
SCOL = ["#b5b5b5", "#a6611a", "#018571", "#7b3294", "#d01c8b"]
DST = ["Paper product", "Cardboard (OCC) bin", "Film bin", "Rigid-plastic bin", "Metal bin"]
POL = [("No QC", "(a) No robotic QC"), ("F1-optimal", "(b) F1-optimal thresholds"),
       ("Value-optimal", "(c) Value-optimal (unconstrained)"), ("Value-optimal, recovery floor", "(d) Value-optimal + recovery floor")]
S = 0.0078          # drawing height per kg (common to all contaminant flows and panels)
STUB = 0.07         # fixed height of the truncated paper -> paper-product flow
GAP, MINH = 0.03, 0.0035
XL, XR, NW = 0.24, 0.70, 0.035

def grade_labels(o):
    g = {2: "SRPN grade", 1: "mixed-paper grade", 0: "off-spec"}[int(o["grade_line"])]
    po = float(o["p_occ"]); occ = "OCC grade" if abs(po - PR["occ"]) < 1e-6 else ("mixed-paper grade" if abs(po - PR["mp"]) < 1e-6 else "off-spec")
    fp = float(o["film_purity"]); film = "Grade A" if fp >= 0.95 else ("Grade B" if fp >= 0.90 else "Grade C")
    return [g, occ, film, None, None]

fig, axs = plt.subplots(2, 2, figsize=(7.48, 8.4)); axs = axs.ravel()
for ax, (pn, title) in zip(axs, POL):
    F, o = flows(pn); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(-0.2, 1.1)
    def h(i, j):
        if i == 0 and j == 0: return STUB
        v = F[i, j]; return 0.0 if v <= 1e-9 else max(MINH, v * S)
    # node extents
    def stack(heights):
        out, y = [], 1.0
        for hh in heights:
            out.append((y - hh, y)); y -= hh + (GAP if hh > 0 else 0)
        return out
    ys = stack([sum(h(i, j) for j in range(5)) for i in range(5)])
    yd = stack([sum(h(i, j) for i in range(5)) for j in range(5)])
    so = [a[1] for a in ys]; do = [a[1] for a in yd]
    for i in range(5):
        for j in range(5):
            hv = h(i, j)
            if hv <= 0: continue
            y0, y1 = so[i], do[j]; so[i] -= hv; do[j] -= hv
            xm = (XL + NW + XR) / 2
            verts = [(XL + NW, y0), (xm, y0), (xm, y1), (XR, y1), (XR, y1 - hv), (xm, y1 - hv), (xm, y0 - hv), (XL + NW, y0 - hv), (XL + NW, y0)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(verts, codes), fc=SCOL[i], alpha=0.35 if (i == 0 and j == 0) else 0.7, lw=0, hatch="////" if (i == 0 and j == 0) else None))
    # source nodes and labels
    for i in range(5):
        a, b = ys[i]
        ax.add_patch(Rectangle((XL, a), NW, b - a, fc=SCOL[i], ec="none"))
        ax.text(XL - 0.015, (a + b) / 2, f"{SRC[i]}  {F[i].sum():.0f}", ha="right", va="center", fontsize=8)
    gl = grade_labels(o)
    labs = []
    for j in range(5):
        a, b = yd[j]; tot = F[:, j].sum()
        if tot <= 1e-9: continue
        ax.add_patch(Rectangle((XR, a), NW, b - a, fc="#3d3d3d", ec="none"))
        lab = f"{DST[j]}  {tot:.1f}" if j else f"{DST[j]}  {tot:.0f}"
        if gl[j]: lab += f"\n{gl[j]}"
        labs.append([(a + b) / 2, lab, (a + b) / 2, 0.095 if gl[j] else 0.055])
    # push labels apart (top to bottom) so they never overlap; draw leader lines when moved
    for k in range(1, len(labs)):
        need = (labs[k - 1][3] + labs[k][3]) / 2
        if labs[k - 1][0] - labs[k][0] < need: labs[k][0] = labs[k - 1][0] - need
    for yl, lab, yn, _ in labs:
        xt = XR + NW + (0.05 if abs(yl - yn) > 1e-6 else 0.015)
        if abs(yl - yn) > 1e-6:
            ax.plot([XR + NW + 0.004, xt - 0.006], [yn, yl], color="#777777", lw=0.6)
        ax.text(xt, yl, lab, ha="left", va="center", fontsize=8, linespacing=1.1)
    if F[:, 1:].sum() <= 1e-9:
        ax.text(XR + NW + 0.015, 0.55, "no bins: nothing\nis picked", ha="left", va="center", fontsize=8, style="italic", color="#555555")
    # break marks on the truncated paper flow
    yb = (ys[0][0] + ys[0][1]) / 2
    ax.text((XL + NW + XR) / 2, yd[0][1] + 0.012, "paper flow truncated", ha="center", va="bottom", fontsize=7.5, style="italic", color="#555555")
    ax.set_title(title, fontsize=9, weight="bold", loc="left", x=0.0)
    # key indicators
    resid = float(o["resid_cb"] + o["resid_nonfibre"])
    txt = (f"Recovered to correct bin: {float(o['recovered_total']):.1f} kg\n"
           f"Misdirected to wrong bin: {float(o['misdirected']):.1f} kg\n"
           f"Contaminants left in paper: {resid:.1f} kg\n"
           f"Paper lost to bins: {float(o['paper_lost']):.1f} kg\n"
           f"Value: {float(o['nsv']):.1f} USD t$^{{-1}}$")
    ax.text(0.0, -0.02, txt, ha="left", va="top", fontsize=8, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc", lw=0.6))
handles = [Line2D([], [], color=c, lw=6, alpha=0.8, label=l) for c, l in zip(SCOL, SRC)]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.005), title="Source material (flow colour)", title_fontsize=8)
fig.text(0.5, -0.005, "All values in kg per tonne of feed. Flow widths are to one common scale, except the hatched paper-to-paper-product flow, which is truncated.",
         ha="center", fontsize=7.5, style="italic")
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.06, wspace=0.12, hspace=0.2)
for ext in ("svg", "pdf"):
    fig.savefig(f"../figures_intermediate/fig9_sankey.{ext}", bbox_inches="tight")
print("sankey ok")
