"""Writes every number used in the manuscript: tables/*.tex and tables/macros.tex (no hand transcription)."""
import json, numpy as np, params as P, model as M
R = json.load(open("../results/results.json")); F3 = json.load(open("../results/fig3_numbers.json"))
FOCAL = "GDINO-B (semi)"; T = "../tables/"
det = R["det_superellipse"]; detp = R["det_power"]; names = list(P.DETECTORS)
G = {2: "SRPN", 1: "MP", 0: "Off-spec"}
def f(x, d=2): return f"{x:.{d}f}"
def tex(n): return n.replace("(sup.)", "(sup.)").replace("_", r"\_")

# ---- Table 2: detector inputs
L = [r"\begin{table}[!htbp]\centering\caption{Detector inputs. mAP and mAP$_{50}$ are reported by \citet{abid2025robust} on the ZeroWaste-f test split; per-class AP$_{50}$ is derived as described in Section~\ref{sec:pr}. Values marked $^\dagger$ borrow the per-class pattern of GDINO-B (sup.).}\label{tab:detectors}",
     r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrr}\toprule",
     r"Detector & mAP (\%) & mAP$_{50}$ (\%) & \multicolumn{4}{c}{Derived AP$_{50,c}$}\\\cmidrule(lr){4-7}",
     r" & & & Cardboard & Soft pl. & Rigid pl. & Metal\\\midrule"]
for n in names:
    ap, mAP, mAP50 = P.DETECTORS[n]; a = M.class_ap50(n); dag = "" if ap is not None else r"$^\dagger$"
    L.append(f"{tex(n)}{dag} & {mAP:.1f} & {mAP50:.1f} & " + " & ".join(f(x) for x in a) + r"\\")
L += [r"\bottomrule\end{tabular}}\end{table}"]
open(T + "tab_detectors.tex", "w").write("\n".join(L))

# ---- Table 3: parameters
pr = P.PRICES["2026-06"]; pr5 = P.PRICES["2025-06"]; B = P.BASE
SRC = dict(p="a", g="b", f="c", d="d", A="A")
rows = [
 ("H", r"Market prices (USD t$^{-1}$; national averages, sorted and baled)"),
 ("Sorted residential papers (PS 56)", f(pr["srpn"]), f(pr5["srpn"]), "a"),
 ("Mixed paper (PS 54)", f(pr["mp"]), f(pr5["mp"]), "a"),
 ("Old corrugated containers (PS 11)", f(pr["occ"]), f(pr5["occ"]), "a"),
 ("PE film, Grade A", f"{pr['film_a']:.1f}", f"{pr5['film_a']:.1f}", "a"),
 ("PE film, Grade B", f"{pr['film_b']:.1f}", f"{pr5['film_b']:.1f}$^\\ast$", "a"),
 ("PE film, Grade C", f"$-${abs(pr['film_c']):.1f}", f"$-${abs(pr5['film_c']):.1f}$^\\ast$", "a"),
 ("Rigid plastics (HDPE proxy)", f(pr["rigid"]), f(pr5["rigid"]), "a"),
 ("Aluminium used beverage cans", f"{pr['ubc']:.0f}", f"{pr5['ubc']:.0f}", "a"),
 ("Steel cans", f"{pr['steel']:.0f}", f"{pr5['steel']:.0f}", "a"),
 ("H", r"Grade rules (maximum share of bale mass)"),
 ("SRPN: prohibitives / outthrows", "2\\% / 3\\%", "--", "b"),
 ("Mixed paper: prohibitives", "2\\%", "--", "b"),
 ("OCC: prohibitives / total non-OCC", "1\\% / 5\\%", "--", "b"),
 ("PE film: Grade A / B contamination", "5\\% / 10\\%", "--", "c"),
 ("H", r"Stream data"),
 ("Item-count mix CB/SP/RP/M (\\%)", "/".join(f"{100*x:.1f}" for x in P.COUNT_SHARE), "--", "d"),
 ("H", r"Modelling assumptions (baseline; range used in sensitivity analysis)"),
 ("Target mass share of feed, $w$", f"{B['w']:.2f}", "0.03--0.15", "A"),
 ("Item mass CB/SP/RP/M (g)", "/".join(f"{1000*x:.0f}" for x in B["m"]), "$\\times$0.5--$\\times$2", "A"),
 ("Paper-piece mass, $m_p$ (g)", f"{1000*B['m_paper']:.0f}", "5--30", "A"),
 ("False-positive share from paper, $\\phi$", f"{B['phi']:.1f}", "0.2--0.8", "A"),
 ("Pick success, $\\eta$", f"{B['eta']:.2f}", "0.85--1.00", "A"),
 ("Aluminium share of metal class", f"{B['al_share']:.1f}", "0.2--0.8", "A"),
 ("Disposal charge on contaminants (USD t$^{-1}$)", f"{B['disposal']:.0f}", "25--100", "A"),
 ("Off-spec paper price (USD t$^{-1}$)", f"{B['p_off']:.0f}", "$-$20--20", "A"),
]
L = [r"\begin{table}[!htbp]\centering", r"\begingroup\linespread{1}\selectfont\small",
     r"\caption{Model parameters and their sources. Prices are converted from \textcent/lb and USD per short ton to USD per metric tonne.}\label{tab:params}",
     r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash\leftskip=0.9em}X ccc@{}}\toprule",
     r"Parameter & Baseline (2026) & 2025 / range & Source\\\midrule"]
first = True
for r_ in rows:
    if r_[0] == "H":
        if not first: L.append(r"\addlinespace[3pt]")
        L.append(r"\multicolumn{4}{@{}l}{\emph{" + r_[1] + r"}}\\"); first = False
    else:
        L.append(f"{r_[0]} & {r_[1]} & {r_[2]} & {r_[3]}" + r"\\")
L += [r"\bottomrule\end{tabularx}",
      r"\par\smallskip{\raggedright\footnotesize Sources: a, \citet{smp2026june}; b, \citet{rema2016paper}; c, \citet{apr2024film}; d, \citet{abid2025robust}; A, modelling assumption, varied in the sensitivity and Monte Carlo analyses. CB, cardboard; SP, soft plastic; RP, rigid plastic; M, metal. $^\ast$Not reported for 2025; held at 2026 values.\par}",
      r"\endgroup\end{table}"]
open(T + "tab_params.tex", "w").write("\n".join(L))

# ---- Table 4: detector results
nso = np.array([det[n]["nsv_opt"] for n in names]); mAP = np.array([P.DETECTORS[n][1] for n in names])
rk_map = (-mAP).argsort().argsort() + 1; rk_v = (-nso).argsort().argsort() + 1
L = [r"\begin{table}[!htbp]\centering\caption{Net sales value (NSV, USD per tonne of paper-line feed) by detector and operating policy; superellipse PR family, baseline parameters. Grade refers to the paper-line output. The no-QC reference is " + f(det[FOCAL]["none"]) + r" USD t$^{-1}$ (mixed-paper grade). Last column: NSV under the power PR family.}\label{tab:results}",
     r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrrr}\toprule",
     r"Detector & NSV$_{F1}$ & Grade$_{F1}$ & NSV$^\ast$ & Grade$^\ast$ & Rank mAP & Rank NSV$^\ast$ & NSV$^\ast$ (power)\\\midrule"]
for i, n in enumerate(names):
    d = det[n]
    L.append(f"{tex(n)} & {f(d['nsv_f1'])} & {G[d['grade_f1']]} & {f(d['nsv_opt'])} & {G[d['grade_opt']]} & {rk_map[i]} & {rk_v[i]} & {f(detp[n]['nsv_opt'])}" + r"\\")
L += [r"\bottomrule\end{tabular}}\end{table}"]
open(T + "tab_results.tex", "w").write("\n".join(L))

# ---- Table 5: budgets
b = R["budget"]
L = [r"\begin{table}[!htbp]\centering\caption{Effect of a pick budget $B$ (attempted picks per tonne of feed) for GDINO-B (semi). F1 policy: all per-class F1 recalls scaled by a common factor until the budget holds. Value-optimal: Eq.~(\ref{eq:opt}).}\label{tab:budget}",
     r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrr}\toprule", r"$B$ & NSV$_{F1}$ & Grade$_{F1}$ & NSV$^\ast$ & Grade$^\ast$ & $\Delta$ & Picks$^\ast$\\\midrule"]
for k in ("inf", "1800", "1400", "1000", "600", "300"):
    d = b[k]; lab = r"$\infty$" if k == "inf" else k
    L.append(f"{lab} & {f(d['nsv_f1'])} & {G[d['grade_f1']]} & {f(d['nsv_opt'])} & {G[d['grade_opt']]} & {f(d['nsv_opt']-d['nsv_f1'])} & {d['picks_opt']:.0f}" + r"\\")
L += [r"\bottomrule\end{tabular}}\end{table}"]
open(T + "tab_budget.tex", "w").write("\n".join(L))

# ---- Table 6: class weights
cw = R["class_weights"]; ap = det[FOCAL]["ap50"]
L = [r"\begin{table}[!htbp]\centering\caption{Economic value of a +0.05 improvement in each class's AP$_{50}$ (GDINO-B (semi), thresholds re-optimised) compared with the equal weight each class receives in mAP.}\label{tab:weights}",
     r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrr}\toprule", r"Class & Baseline AP$_{50}$ & $\Delta$NSV$^\ast$ (USD t$^{-1}$) & Economic weight (\%) & Weight in mAP (\%)\\\midrule"]
for c in range(4):
    L.append(f"{P.LABELS[c]} & {f(ap[c])} & {f(cw['dv'][c],3)} & {100*cw['share'][c]:.1f} & 25.0" + r"\\")
L += [r"\bottomrule\end{tabular}}\end{table}"]
open(T + "tab_weights.tex", "w").write("\n".join(L))

# ---- Table 7: Monte Carlo
mc = R["mc"]
def q(a): a = np.array(a); return f(a.mean()), f(np.percentile(a, 5)), f(np.median(a)), f(np.percentile(a, 95))
L = [r"\begin{table}[!htbp]\centering\caption{Monte Carlo summary (" + str(len(mc["tau"])) + r" draws; all assumptions in Table~\ref{tab:params} sampled jointly, price year and PR family drawn at random). Values in USD t$^{-1}$ except $\tau$.}\label{tab:mc}",
     r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrr}\toprule", r"Quantity & Mean & P5 & Median & P95\\\midrule"]
for lab, key in (("Value of QC, F1 thresholds", "gain_f1_none"), ("Value of QC, value-optimal thresholds", "gain_opt_none"),
                 ("Gain of value-optimal over F1 thresholds", "gain_opt_f1"), ("Kendall $\\tau$ (mAP vs.\\ NSV$^\\ast$ rank)", "tau")):
    L.append(lab + " & " + " & ".join(q(mc[key])) + r"\\")
L += [r"\bottomrule\end{tabular}}\end{table}"]
open(T + "tab_mc.tex", "w").write("\n".join(L))

# ---- macros
fd = det[FOCAL]; tor = R["tornado"]
gopt_f1 = np.array(mc["gain_opt_f1"]); gf1 = np.array(mc["gain_f1_none"])
Mc = dict(
 NoQC=f(fd["none"]), FocalFone=f(fd["nsv_f1"]), FocalOpt=f(fd["nsv_opt"]),
 GainFone=f(fd["nsv_f1"]-fd["none"]), GainOpt=f(fd["nsv_opt"]-fd["none"]), GainOptFone=f(fd["nsv_opt"]-fd["nsv_f1"]),
 CrossRecall=f(F3["cross"]), CliffJump=f(F3["nsv_at_cross"]-F3["nsv_before"]), NSVBeforeCliff=f(F3["nsv_before"]), NSVAtCliff=f(F3["nsv_at_cross"]),
 SRPNPrem=f(P.PRICES["2026-06"]["srpn"]-P.PRICES["2026-06"]["mp"]),
 WCB=f"{100*R['class_weights']['share'][0]:.0f}", WSP=f"{100*R['class_weights']['share'][1]:.0f}",
 WRP=f"{100*R['class_weights']['share'][2]:.0f}", WM=f"{100*R['class_weights']['share'][3]:.0f}",
 TauBase=f(R["rank_superellipse"]["tau_map_opt"]), TauFone=f(R["rank_superellipse"]["tau_map_f1"]), TauPower=f(R["rank_power"]["tau_map_opt"]),
 TauMed=f(np.median(mc["tau"])), TopShare=f"{100*np.mean(mc['top_is_map_top']):.0f}",
 MCGainMed=f(np.median(gopt_f1)), MCGainLo=f(np.percentile(gopt_f1,5)), MCGainHi=f(np.percentile(gopt_f1,95)),
 MCGainPos=f"{100*np.mean(gopt_f1>0.5):.0f}", MCSRPNFone=f"{100*np.mean(mc['srpn_f1']):.0f}", MCSRPNOpt=f"{100*np.mean(mc['srpn_opt']):.0f}",
 MCFoneNeg=f"{100*np.mean(gf1<0):.0f}", NMC=str(len(mc["tau"])),
 BudF=f(b["1400"]["nsv_f1"]), BudO=f(b["1400"]["nsv_opt"]), BudD=f(b["1400"]["nsv_opt"]-b["1400"]["nsv_f1"]),
 TridF=f(det["TridentNet"]["nsv_f1"]), TridO=f(det["TridentNet"]["nsv_opt"]),
 BestOpt=f(max(d["nsv_opt"] for d in det.values())), SpreadModern=f(max(det[n]["nsv_opt"] for n in names[:7]) - min(det[n]["nsv_opt"] for n in names[:7])),
 FilmPurFone=f"{100*fd['film_pur_f1']:.0f}", PlateauHalf=f"{100*R['plateau']['share_within_0p5']:.1f}", PlateauSRPN=f"{100*R['plateau']['share_srpn']:.0f}",
 ItemsPerT=f"{sum(fd['n']):.0f}", PicksFone=f"{fd['picks_f1']:.0f}",
 WLoGain=f(tor["runs"]["w"][0][2]-tor["runs"]["w"][0][0]), WHiGain=f(tor["runs"]["w"][1][2]-tor["runs"]["w"][1][0]),
 PriceGain=f(tor["runs"]["price"][0][2]-tor["runs"]["price"][0][0]),
 BaseGain=f(tor["base"][2]-tor["base"][0]),
)
# decomposition of focal opt
dec = fd["dec_opt"]; Mc.update(DecLine=f(dec[0]), DecOther=f(sum(dec[1:])), DecLineShare=f"{100*(dec[0]-fd['none'])/(fd['nsv_opt']-fd['none']):.0f}")
open(T + "macros.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in Mc.items()) + "\n")
print("\n".join(f"{k}={v}" for k, v in Mc.items()))
