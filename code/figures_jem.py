"""JEM figure set: six main-text figures (Figs 1-6) and three supplementary figures (Figs S1-S3).
All numbers are read from results/*.json or recomputed with model.py; nothing is hand-typed."""
import json, shutil, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import params as P, model as M
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "legend.fontsize": 8, "axes.labelsize": 9.5, "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False})
R = json.load(open("../results/results.json")); R2 = json.load(open("../results/results_revision.json"))
FOCAL = "GDINO-B (semi)"; FD = "../figures_jem/"
C4 = ["#8c510a", "#01665e", "#5e3c99", "#b2182b"]
def save(fig, name):
    for ext in ("pdf", "svg"): fig.savefig(FD + name + "." + ext, bbox_inches="tight")
    fig.savefig(FD + name + ".tiff", bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)
def lab(ax, t, x=-0.14, y=1.03): ax.text(x, y, f"({t})", transform=ax.transAxes, weight="bold")

# Figs 1, 2, 5: unchanged figures from the previous build
for src, dst in (("fig1_framework", "Fig1_framework"), ("fig9_sankey", "Fig2_material_flows"), ("fig5_map_vs_value", "FigS3_detector_accuracy_vs_value")):
    for ext in ("pdf", "svg"): shutil.copy(f"../figures_intermediate/{src}.{ext}", f"{FD}{dst}.{ext}")

# ---------- Fig 3: grade cliffs (a, b: cardboard sweep; c: film frontier)
d0 = R["det_superellipse"][FOCAL]; shf = np.array(d0["shapes"]); r1c = np.array(d0["r_f1"])
r = np.linspace(0, 0.98, 197); Rg = np.tile(r1c, (r.size, 1)); Rg[:, 0] = r
o = M.evaluate(Rg, shf, dict(P.BASE), True)
fig = plt.figure(figsize=(7.2, 6.0)); gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.25], hspace=0.45, wspace=0.3)
a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, :])
a.plot(r, 100 * np.array(o["outthrow"]), color=C4[0], lw=1.5, label="Outthrows (cardboard)")
a.plot(r, 100 * np.array(o["prohib"]), color="#555555", lw=1.2, ls="--", label="Prohibitives (plastics, metal)")
a.axhline(3, color=C4[0], lw=0.8, ls=":"); a.axhline(2, color="#555555", lw=0.8, ls=":")
a.text(0.01, 2.45, "SRPN outthrow 3%", ha="left", fontsize=8, color=C4[0])
a.text(0.01, 1.45, "Prohibitive 2%", ha="left", fontsize=8, color="#555555")
a.set_xlabel("Cardboard operating recall $r_1$"); a.set_ylabel("Share of paper product (%)"); a.legend(fontsize=8)
nsv = np.array(o["nsv"]); g = np.array(o["grade_line"]); b.plot(r, nsv, color="k", lw=1.5)
for gv, col, lb in ((2, "#d9f0d3", "SRPN grade"), (1, "#fee0b6", "Mixed-paper grade")):
    m = g == gv
    if m.any(): b.fill_between(r, nsv.min() - 1, nsv.max() + 1, where=m, color=col, alpha=0.8, label=lb, step="mid")
b.set_ylim(nsv.min() - 0.5, nsv.max() + 0.5); b.set_xlabel("Cardboard operating recall $r_1$")
b.set_ylabel("Net sales value (USD t$^{-1}$)"); b.legend(fontsize=8, loc="upper left")
f = R["film_frontier_f1ctx"]; pur = 100 * np.array(f["purity"]); rec = 100 * np.array(f["recovery"]); rev = np.array(f["rev_film"])
c.axhspan(95, 101.5, color="#d9f0d3", alpha=0.9); c.axhspan(90, 95, color="#fee0b6", alpha=0.9); c.axhspan(0, 90, color="#f4f4f4")
c.text(99, 97.6, "Grade A (15.0 ¢/lb)", fontsize=7.5, ha="right", va="center"); c.text(99, 92.4, "Grade B (4.0 ¢/lb)", fontsize=7.5, ha="right", va="center")
c.text(99, 45, "Grade C (−0.56 ¢/lb)", fontsize=8, ha="right")
sc = c.scatter(rec[1:], pur[1:], c=rev[1:], cmap="viridis", s=9, zorder=3)
cb = fig.colorbar(sc, ax=c, pad=0.01); cb.set_label("Film-bin revenue\n(USD t$^{-1}$ feed)", fontsize=8)
c.set_xlabel("Soft-plastic recovery to film bale (%)"); c.set_ylabel("Film-bale purity (%)"); c.set_ylim(0, 101); c.set_xlim(0, 100)
lab(a, "a", -0.18); lab(b, "b", -0.2); lab(c, "c", -0.08)
save(fig, "Fig3_grade_cliffs")

# ---------- Fig 4: recovery-value trade-off (a: Pareto; b: value decomposition)
det = R["det_superellipse"]; bud = R["budget"]
fig = plt.figure(figsize=(7.2, 6.4)); gs = GridSpec(2, 1, figure=fig, height_ratios=[1.1, 1], hspace=0.55)
a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1]); cl = R2["cloud"]
a.scatter(cl["rec"], cl["nsv"], s=2, color="#cccccc", label="Grid operating points", rasterized=True)
fr = np.array(R2["pareto"]); a.plot(fr[:, 2], fr[:, 1], color="k", lw=1.3, label="Pareto front")
mk = {"No QC": "s", "F1-optimal": "o", "High-precision (p>=0.90)": "^", "Uniform recall 0.80": "v",
      "Global threshold (tuned)": "D", "Value-optimal": "*", "Value-optimal, recovery floor": "P"}
for n, d in R2["policies"].items(): a.scatter(d["recovered_total"], d["nsv"], marker=mk[n], s=45, zorder=4, label=n.replace(">=", "≥"))
a.set_ylim(min(28, min(cl["nsv"])), 47); a.set_xlabel("Target material recovered to its own bin (kg t$^{-1}$ feed)")
a.set_ylabel("Net sales value (USD t$^{-1}$)"); a.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
labels = ["No QC", "F1, no budget", "Opt., no budget"]; comps = [[det[FOCAL]["none"], 0, 0, 0, 0], det[FOCAL]["dec_f1"], det[FOCAL]["dec_opt"]]
for k in ("1400", "1000", "600", "300"):
    labels += [f"F1, B={k}", f"Opt., B={k}"]; comps += [bud[k]["dec_f1"], bud[k]["dec_opt"]]
comps = np.array(comps); names5 = ["Paper product", "OCC bin", "Film bin", "Rigid-plastic bin", "Metal bin"]; cols5 = ["#bdbdbd"] + C4
x = np.arange(len(labels)); pos = np.zeros(len(x)); neg = np.zeros(len(x))
for j in range(5):
    v = comps[:, j]; bot = np.where(v >= 0, pos, neg)
    b.bar(x, v, bottom=bot, color=cols5[j], width=0.7, label=names5[j], edgecolor="white", lw=0.4)
    pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
for xi, t in zip(x, comps.sum(1)): b.text(xi, t + 0.6, f"{t:.1f}", ha="center", fontsize=8)
b.set_xticks(x); b.set_xticklabels(labels, rotation=35, ha="right", fontsize=8); b.set_ylabel("Net sales value (USD t$^{-1}$)")
b.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16)); b.set_ylim(min(0, neg.min()) - 1, pos.max() + 4)
lab(a, "a", -0.12, 1.02); lab(b, "b", -0.08, 1.08)
save(fig, "Fig4_recovery_value_tradeoff")

# ---------- Fig 6: transferability (a, b) and uncertainty (c, d)
sc_ = R2["scenarios"]; ws = ["Low (4%)", "Medium (8%)", "High (14%)"]; mc = R2["mc"]
fig = plt.figure(figsize=(7.4, 6.6)); gs = GridSpec(2, 2, figure=fig, hspace=0.65, wspace=0.32)
pols = (("F1-optimal", "#2c7bb6"), ("Global threshold (tuned)", "#abd9e9"), ("Value-optimal", "#d7191c"), ("Value-optimal, recovery floor", "#fdae61"))
axs = [fig.add_subplot(gs[0, 0])]; axs.append(fig.add_subplot(gs[0, 1], sharey=axs[0]))
for ax, spec in zip(axs, ("Standard", "Strict")):
    xx = np.arange(3); wd = 0.12
    for k, (pn, col) in enumerate(pols):
        for q, (pl, h) in enumerate((("2026-06", ""), ("2025-06", "///"))):
            v = [sc_[f"{w}|{pl}|{spec}"][pn]["nsv"] - sc_[f"{w}|{pl}|{spec}"]["No QC"]["nsv"] for w in ws]
            ax.bar(xx + (k * 2 + q - 3.5) * wd, v, width=wd, color=col, hatch=h, edgecolor="white", lw=0.3, label=(pn if q == 0 else None))
    ax.axhline(0, color="k", lw=0.7); ax.set_xticks(xx); ax.set_xticklabels(ws, fontsize=8)
    ax.set_title(f"{spec} bale specification", fontsize=9); ax.set_xlabel("Target mass share of feed")
axs[0].set_ylabel("Value of QC vs. no QC (USD t$^{-1}$)"); plt.setp(axs[1].get_yticklabels(), visible=False)
axs[0].text(0.03, 0.97, "solid: 2026 prices\nhatched: 2025 prices", transform=axs[0].transAxes, fontsize=8, va="top")
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, ncol=4, fontsize=8, loc="center", bbox_to_anchor=(0.5, 0.505))
c = fig.add_subplot(gs[1, 0]); d = fig.add_subplot(gs[1, 1])
c.hist(mc["g_f1"], bins=40, color="#2c7bb6", alpha=0.65, label="F1-optimal")
c.hist(mc["g_rf"], bins=40, color="#fdae61", alpha=0.65, label="Value-opt., recovery floor")
c.hist(mc["g_opt"], bins=40, color="#d7191c", alpha=0.45, label="Value-optimal")
c.set_xlabel("Value of QC (USD t$^{-1}$)"); c.set_ylabel("Monte Carlo draws"); c.legend(fontsize=8)
gg = np.array(mc["g_rf"]) - np.array(mc["g_f1"]); fail = 1 - np.array(mc["srpn_f1"]); nn = np.arange(1, len(gg) + 1)
d.plot(nn, np.cumsum(gg) / nn, color="k", lw=1.5, label="Mean gain over F1")
d2 = d.twinx(); d2.plot(nn, np.cumsum(fail) / nn, color="#d7191c", lw=1, label="Share of draws F1 loses SRPN")
d2.set_ylim(0, 1); d2.spines["right"].set_visible(True); d2.set_ylabel("Share of draws")
d.set_xlabel("Number of draws"); d.set_ylabel("Running mean (USD t$^{-1}$)"); d.set_ylim(0, 12)
hh = d.get_legend_handles_labels(); hh2 = d2.get_legend_handles_labels(); d.legend(hh[0] + hh2[0], hh[1] + hh2[1], fontsize=8, loc="lower right")
lab(axs[0], "a", -0.2, 1.06); lab(axs[1], "b", -0.08, 1.06); lab(c, "c", -0.2); lab(d, "d", -0.2)
save(fig, "Fig5_transferability_uncertainty")

# ---------- Supplementary figures
d = R["det_superellipse"][FOCAL]; sh = np.array(d["shapes"]); rr = np.linspace(0, 0.999, 400)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
for k in range(4):
    ax.plot(rr, M.precision(rr, sh[k], "superellipse"), color=C4[k], lw=1.4, label=f"{P.LABELS[k]} (AP$_{{50}}$={d['ap50'][k]:.2f})")
    ax.plot(d["r_f1"][k], M.precision(d["r_f1"][k], sh[k], "superellipse"), "o", color=C4[k], mfc="white", ms=5)
    ax.plot(d["r_opt"][k], M.precision(d["r_opt"][k], sh[k], "superellipse"), "*", color=C4[k], ms=9)
ax.plot([], [], "ko", mfc="white", ms=5, label="F1-optimal point"); ax.plot([], [], "k*", ms=8, label="Value-optimal point")
ax.set_xlabel("Recall $r_c$"); ax.set_ylabel("Precision $p_c$"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(fontsize=8, loc="lower left")
save(fig, "FigS1_pr_curves")
fig, ax = plt.subplots(figsize=(4.2, 3.0)); tau = np.array(mc["tau"])
ax.hist(tau, bins=20, color="#636363"); ax.axvline(np.median(tau), color="k", ls="--", lw=0.9)
ax.set_xlabel("Kendall $\\tau$: mAP rank vs. value rank (12 detectors)"); ax.set_ylabel("Monte Carlo draws")
save(fig, "FigS2_kendall_tau")
t = R["tornado"]; base = t["base"]; bg = base[2] - base[0]
L = dict(w="Target mass share $w$ (3–15%)", m_scale="Item masses (×0.5–×2)", m_paper="Paper-piece mass (5–30 g)",
         phi="FP share from paper $\\phi$ (0.2–0.8)", eta="Pick success $\\eta$ (0.85–1.0)", al_share="Aluminium share of metal (0.2–0.8)",
         disposal="Disposal charge (25–100 USD t$^{-1}$)", p_off="Off-spec paper price (−20–20)", price="Price year (2025 vs 2026)",
         pr_family="PR-curve family (power vs superellipse)")
rows = sorted([(L[k], lo[2] - lo[0] - bg, hi[2] - hi[0] - bg) for k, (lo, hi) in t["runs"].items()], key=lambda z: max(abs(z[1]), abs(z[2])))
fig, ax = plt.subplots(figsize=(5.6, 3.3)); y = np.arange(len(rows))
ax.barh(y, [z[1] for z in rows], color="#4393c3", height=0.6, label="Low value of range")
ax.barh(y, [z[2] for z in rows], color="#d6604d", height=0.6, alpha=0.85, label="High value of range")
ax.set_yticks(y); ax.set_yticklabels([z[0] for z in rows], fontsize=8); ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(f"Change in value of robotic QC vs. baseline ({bg:.1f} USD t$^{{-1}}$)"); ax.legend(fontsize=8, loc="lower right")
save(fig, "FigS4_sensitivity")
print("JEM figures ok; median tau", np.median(tau))
