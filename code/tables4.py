"""Revision tables and macros: GL vs VO-RF gap decomposition (results_caie_gap.json) and
sensitivity of structural results to the PR-curve family (fresh model evaluations)."""
import json, numpy as np, params as P, model as M
from decimal import Decimal, ROUND_HALF_UP
T = "../tables/"
def f(x, d=1):
    t = str(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
    return ("$-$" + t[1:]) if t.startswith("-") and float(t) != 0 else t.lstrip("-")
G = json.load(open("../results/results_caie_gap.json")); a = lambda k: np.array(G[k], float)
GL, RF, VO = a("gain_GL"), a("gain_RF"), a("gain_VO"); fb = a("floor_binds").astype(bool); fbk = a("rf_fallback").astype(bool)
w = a("w"); recGL, recRF = a("rec_GL"), a("rec_RF")
sub = [(i, x) for i, x in enumerate(G["gain_RF30"]) if x is not None]; idx = np.array([i for i, _ in sub]); v30 = np.array([x for _, x in sub])

# ---------------- table: gap decomposition by contamination level
bands = ((0.03, 0.06), (0.06, 0.09), (0.09, 0.12), (0.12, 0.15 + 1e-9))
rows = []
def row(lab, m):
    return (f"{lab} & {int(m.sum()):,} & {f(GL[m].mean(),2)} & {f(RF[m].mean(),2)} & {f((VO-GL)[m].mean(),2)} & {f((VO-RF)[m].mean(),2)} & "
            f"{f((RF-GL)[m].mean(),2)} & {f(100*fb[m].mean(),0)} & {f((recRF-recGL)[m].mean())} & {f(100*(RF[m]<0).mean(),1)}" + r"\\")
for lo, hi in bands:
    rows.append(row(f"{f(100*lo,0)}--{f(100*min(hi,0.15),0)}\\%", (w >= lo) & (w < hi)))
rows.append(r"\midrule " + row("All draws", np.ones_like(w, bool)))
open(T + "tab_gap_decomp.tex", "w").write("\n".join([
    r"\begin{table}[!htbp]\centering", r"\begingroup\linespread{1}\selectfont\footnotesize\setlength{\tabcolsep}{3pt}",
    r"\caption{Why the tuned global threshold (GL) has a higher mean value than VO-RF in the paired Monte Carlo analysis, by target mass share $w$ of the feed (" + f"{len(w):,}" + r" draws; means; values of QC in USD t$^{-1}$).}\label{tab:gapdecomp}",
    r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}\toprule",
    r"\shortstack[l]{Target\\share $w$} & Draws & GL & VO-RF & \shortstack{Flexi-\\bility} & \shortstack{Floor\\cost} & Net & \shortstack{Floor\\binds\\(\% draws)} & \shortstack{Extra\\recovery\\(kg t$^{-1}$)} & \shortstack{VO-RF $<$\\no QC\\(\% draws)}\\\midrule"] + rows +
    [r"\bottomrule\end{tabular*}",
     r"\par\smallskip{\raggedright\footnotesize Flexibility $=$ VO $-$ GL: gain of unconstrained per-class thresholds (VO) over GL. Floor cost $=$ VO $-$ VO-RF: loss from the recovery floor and the film constraint. Net $=$ VO-RF $-$ GL $=$ flexibility $-$ floor cost. Floor binds: share of draws in which the VO solution violates the recovery floor or the film constraint. Extra recovery: VO-RF minus GL. VO-RF $<$ no QC: share of draws in which VO-RF is worth less than no quality control.\par}",
     r"\endgroup", r"\end{table}"]))

# ---------------- PR-family sensitivity of the structural results (baseline, focal detector)
pr = {}
for fam in ("superellipse", "power"):
    prm = dict(P.BASE, pr_family=fam); sh = M.shapes_for("GDINO-B (semi)", fam)
    r1 = M.f1_policy(sh, fam); g = np.linspace(0, 0.98, 4901); Rg = np.tile(r1, (g.size, 1)); Rg[:, 0] = g
    o = M.evaluate(Rg, sh, prm, True); j = int(np.argmax(o["grade_line"] == 2))
    _, vo = M.optimize(sh, prm, k=50)
    pr[fam] = dict(rhat=g[j], jump=float(o["nsv"][j] - o["nsv"][j - 1]), r1cb=float(r1[0]), vo=vo, f1=float(M.evaluate(r1, sh, prm)))

mac = dict(
    GapFlex=f((VO - GL).mean(), 2), GapFlexShare=f(100 * np.mean(VO >= GL - 1e-9), 0), GapFloorCost=f((VO - RF).mean(), 2),
    GapNet=f((GL - RF).mean(), 2), FloorBindShare=f(100 * fb.mean(), 0), FloorFreeGain=f((RF - GL)[~fb].mean(), 2),
    FloorFreeShare=f(100 * (~fb).mean(), 0), FallbackN=str(int(fbk.sum())), FallbackLoss=f((GL - RF)[fbk].mean(), 2),
    RFthirtyGain=f((v30 - RF[idx]).mean(), 2), RFthirtyNet=f((GL[idx] - v30).mean(), 2), RFthirtyN=str(len(idx)),
    LowWNet=f((GL - RF)[(w < 0.06)].mean(), 2), LowWNeg=f(100 * (RF[w < 0.06] < 0).mean(), 0), LowWRec=f((recRF - recGL)[w < 0.06].mean()),
    MidWNet=f((RF - GL)[(w >= 0.09) & (w < 0.12)].mean(), 2), NegBinding=f(100 * np.mean(fb[RF < 0]), 0),
    PRpowVO=f(pr["power"]["vo"], 2), PRseVO=f(pr["superellipse"]["vo"], 2), PRdiffVO=f(pr["power"]["vo"] - pr["superellipse"]["vo"], 2),
    PRpowRhat=f(pr["power"]["rhat"], 3), PRseRhat=f(pr["superellipse"]["rhat"], 3), PRpowJump=f(pr["power"]["jump"], 2), PRseJump=f(pr["superellipse"]["jump"], 2),
    PRpowFOneCB=f(pr["power"]["r1cb"], 3), PRseFOneCB=f(pr["superellipse"]["r1cb"], 3),
    PRpowMargin=f(pr["power"]["r1cb"] - pr["power"]["rhat"], 2), PRseMargin=f(pr["superellipse"]["r1cb"] - pr["superellipse"]["rhat"], 2),
)
for k in mac: assert k.isalpha(), k
open(T + "macros4.tex", "w").write("\n".join(r"\newcommand{\%s}{%s}" % (k, v) for k, v in mac.items()) + "\n")
print(mac)
