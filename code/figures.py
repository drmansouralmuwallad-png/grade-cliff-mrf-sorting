import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import params as P, model as M

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8, "axes.labelsize": 9.5, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
                     "pdf.fonttype": 42, "legend.frameon": False})
C4 = ["#8c510a", "#01665e", "#5e3c99", "#b2182b"]   # cardboard, soft, rigid, metal
R = json.load(open("../results/results.json"))
FOCAL = "GDINO-B (semi)"
FD = "../figures_intermediate/"
def save(fig, name):
    fig.savefig(FD + name + ".svg", bbox_inches="tight"); fig.savefig(FD + name + ".pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 2 calibrated PR curves
d = R["det_superellipse"][FOCAL]; sh = np.array(d["shapes"])
fig, ax = plt.subplots(figsize=(3.6, 3.2)); rr = np.linspace(0, 0.999, 400)
for c in range(4):
    ax.plot(rr, M.precision(rr, sh[c], "superellipse"), color=C4[c], lw=1.4, label=f"{P.LABELS[c]} (AP$_{{50}}$={d['ap50'][c]:.2f})")
    r1 = d["r_f1"][c]; ax.plot(r1, M.precision(r1, sh[c], "superellipse"), "o", color=C4[c], mfc="white", ms=5)
    ro = d["r_opt"][c]; ax.plot(ro, M.precision(ro, sh[c], "superellipse"), "*", color=C4[c], ms=9)
ax.plot([], [], "ko", mfc="white", ms=5, label="F1-optimal point"); ax.plot([], [], "k*", ms=8, label="Value-optimal point")
ax.set_xlabel("Recall $r_c$"); ax.set_ylabel("Precision $p_c$"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(fontsize=8, loc="lower left"); save(fig, "fig2_pr_curves")

# ---------------- Fig 3 cardboard sweep
d0 = R["det_superellipse"][FOCAL]; shf = np.array(d0["shapes"]); r1c = np.array(d0["r_f1"])
r = np.linspace(0, 0.98, 197); Rg = np.tile(r1c, (r.size, 1)); Rg[:, 0] = r
_o = M.evaluate(Rg, shf, dict(P.BASE), True)
s = dict(outthrow=_o["outthrow"], prohib=_o["prohib"], nsv=_o["nsv"], grade=_o["grade_line"])
cross = r[np.argmax(_o["grade_line"] == 2)]; json.dump(dict(cross=float(cross), nsv_at_cross=float(_o["nsv"][np.argmax(_o["grade_line"] == 2)]), nsv_before=float(_o["nsv"][np.argmax(_o["grade_line"] == 2) - 1])), open("../results/fig3_numbers.json", "w"))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
axs[0].plot(r, 100 * np.array(s["outthrow"]), color=C4[0], lw=1.5, label="Outthrows (cardboard) in paper line")
axs[0].plot(r, 100 * np.array(s["prohib"]), color="#555555", lw=1.2, ls="--", label="Prohibitives (plastics, metal)")
axs[0].axhline(3, color=C4[0], lw=0.8, ls=":"); axs[0].axhline(2, color="#555555", lw=0.8, ls=":")
axs[0].text(0.99, 3.15, "SRPN outthrow limit 3%", ha="right", fontsize=8, color=C4[0])
axs[0].text(0.99, 1.4, "SRPN/MP prohibitive limit 2%", ha="right", fontsize=8, color="#555555")
axs[0].set_xlabel("Cardboard operating recall $r_1$"); axs[0].set_ylabel("Share of paper-line output (%)"); axs[0].legend(fontsize=8)
nsv = np.array(s["nsv"]); axs[1].plot(r, nsv, color="k", lw=1.5)
g = np.array(s["grade"]); 
for gv, col, lab in ((2, "#d9f0d3", "SRPN grade"), (1, "#fee0b6", "Mixed paper grade")):
    msk = g == gv
    if msk.any(): axs[1].fill_between(r, nsv.min() - 1, nsv.max() + 1, where=msk, color=col, alpha=0.8, label=lab, step="mid")
axs[1].set_ylim(nsv.min() - 0.5, nsv.max() + 0.5)
axs[1].set_xlabel("Cardboard operating recall $r_1$"); axs[1].set_ylabel("Net sales value (USD t$^{-1}$ feed)"); axs[1].legend(fontsize=8, loc="center right")
for a, t in zip(axs, "ab"): a.text(-0.16, 1.02, f"({t})", transform=a.transAxes, weight="bold")
fig.tight_layout(); save(fig, "fig3_cardboard_cliff")

# ---------------- Fig 4 film purity-recovery frontier
f = R["film_frontier_f1ctx"]; pur = 100 * np.array(f["purity"]); rec = 100 * np.array(f["recovery"]); rev = np.array(f["rev_film"])
fig, ax = plt.subplots(figsize=(3.8, 3.2))
ax.axhspan(95, 101, color="#d9f0d3", alpha=0.9); ax.axhspan(90, 95, color="#fee0b6", alpha=0.9); ax.axhspan(0, 90, color="#f4f4f4")
ax.text(30, 96.1, "Grade A (15.0 ¢/lb)", fontsize=8); ax.text(30, 91.1, "Grade B (4.0 ¢/lb)", fontsize=8); ax.text(40, 60, "Grade C (−0.56 ¢/lb)", fontsize=8)
sc = ax.scatter(rec[1:], pur[1:], c=rev[1:], cmap="viridis", s=9, zorder=3)
cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("Film-bin revenue (USD t$^{-1}$ feed)", fontsize=8)
ax.set_xlabel("Soft-plastic recovery to film bale (%)"); ax.set_ylabel("Film-bale purity (%)"); ax.set_ylim(max(0, pur[1:].min() - 5), 100.5); ax.set_xlim(0, 100)
save(fig, "fig4_film_frontier")

# ---------------- Fig 5 is produced by fig5_map_vs_value.py
exec(open("fig5_map_vs_value.py").read())

# ---------------- Fig 6 revenue decomposition by policy / budget
b = R["budget"]; labels = ["No QC"]; comps = [[det[FOCAL]["none"], 0, 0, 0, 0]]
labels.append("F1, no budget"); comps.append(det[FOCAL]["dec_f1"]); labels.append("Opt., no budget"); comps.append(det[FOCAL]["dec_opt"])
for key in ("1400", "1000", "600", "300"):
    labels.append(f"F1, B={key}"); comps.append(b[key]["dec_f1"]); labels.append(f"Opt., B={key}"); comps.append(b[key]["dec_opt"])
comps = np.array(comps); names5 = ["Paper line", "OCC bin", "Film bin", "Rigid-plastic bin", "Metal bin"]; cols5 = ["#bdbdbd", C4[0], C4[1], C4[2], C4[3]]
fig, ax = plt.subplots(figsize=(7.2, 3.0)); x = np.arange(len(labels)); pos = np.zeros(len(labels)); neg = np.zeros(len(labels))
for j in range(5):
    v = comps[:, j]; bot = np.where(v >= 0, pos, neg)
    ax.bar(x, v, bottom=bot, color=cols5[j], width=0.7, label=names5[j], edgecolor="white", lw=0.4)
    pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
for xi, tot in zip(x, comps.sum(1)): ax.text(xi, tot + 0.6, f"{tot:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8); ax.set_ylabel("Net sales value (USD t$^{-1}$ feed)")
ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16)); ax.set_ylim(min(0, neg.min()) - 1, pos.max() + 4)
save(fig, "fig6_value_decomposition")

# ---------------- Fig 7 tornado (gain of value-optimal over no QC, and over F1)
t = R["tornado"]; base = t["base"]; bg = base[2] - base[0]
lab = dict(w="Target mass share $w$ (3–15%)", m_scale="Item masses (×0.5–×2)", m_paper="Paper-piece mass (5–30 g)",
           phi="FP share from paper $\\phi$ (0.2–0.8)", eta="Pick success $\\eta$ (0.85–1.0)", al_share="Aluminium share of metal (0.2–0.8)",
           disposal="Disposal charge (25–100 USD t$^{-1}$)", p_off="Off-spec paper price (−20–20)", price="Price year (2025 vs 2026)",
           pr_family="PR-curve family (power vs superellipse)")
rows = []
for k_, (lo, hi) in t["runs"].items():
    rows.append((lab[k_], lo[2] - lo[0] - bg, hi[2] - hi[0] - bg))
rows.sort(key=lambda z: max(abs(z[1]), abs(z[2])))
fig, ax = plt.subplots(figsize=(5.4, 3.2)); y = np.arange(len(rows))
ax.barh(y, [r_[1] for r_ in rows], color="#4393c3", height=0.6, label="Low value of range")
ax.barh(y, [r_[2] for r_ in rows], color="#d6604d", height=0.6, alpha=0.85, label="High value of range")
ax.set_yticks(y); ax.set_yticklabels([r_[0] for r_ in rows], fontsize=8); ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(f"Change in value of robotic QC vs. baseline ({bg:.1f} USD t$^{{-1}}$)"); ax.legend(fontsize=8, loc="lower right")
save(fig, "fig7_tornado")

# ---------------- Fig 8 Monte Carlo
mc = R["mc"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
axs[0].hist(mc["gain_f1_none"], bins=30, color="#2c7bb6", alpha=0.7, label="F1 thresholds vs. no QC")
axs[0].hist(mc["gain_opt_none"], bins=30, color="#d7191c", alpha=0.55, label="Value-optimal vs. no QC")
axs[0].set_xlabel("Value of robotic QC (USD t$^{-1}$ feed)"); axs[0].set_ylabel("Monte Carlo draws"); axs[0].legend(fontsize=8)
axs[1].hist(mc["tau"], bins=20, color="#636363"); axs[1].axvline(np.median(mc["tau"]), color="k", ls="--", lw=0.9)
axs[1].set_xlabel("Kendall $\\tau$: mAP rank vs. value rank (12 detectors)"); axs[1].set_ylabel("Monte Carlo draws")
for a, tt in zip(axs, "ab"): a.text(-0.16, 1.02, f"({tt})", transform=a.transAxes, weight="bold")
fig.tight_layout(); save(fig, "fig8_monte_carlo")
print("figures ok")
