"""CAIE tables and macros, written from results_caie.json, results_revision.json and results.json.
Nothing is hand-typed: every number in the CAIE text, tables and macros comes from these files."""
import json, re, numpy as np, params as P, model as M
R = json.load(open("../results/results.json")); R2 = json.load(open("../results/results_revision.json"))
R3 = json.load(open("../results/results_caie.json")); T = "../tables/"
G = {2: "SRPN", 1: "MP", 0: "Off-spec"}; GO = {2: "OCC", 1: "MP", 0: "Off-spec"}; GF = {2: "A", 1: "B", 0: "C"}
def f(x, d=1):
    from decimal import Decimal, ROUND_HALF_UP
    t = str(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
    return ("$-$" + t[1:]) if t.startswith("-") and float(t) != 0 else t.lstrip("-")
def wrap(cap, lab, colspec, head, rows, notes=None, resize=True):
    body = [r"\begin{table}[!htbp]\centering\caption{" + cap + r"}\label{" + lab + "}"]
    body += [(r"\resizebox{\textwidth}{!}{" if resize else "") + r"\begin{tabular}{" + colspec + r"}\toprule", head + r"\\\midrule"]
    body += rows + [r"\bottomrule\end{tabular}" + ("}" if resize else "")]
    if notes: body += [r"\par\smallskip{\raggedright\footnotesize " + notes + r"\par}"]
    return "\n".join(body + [r"\end{table}"])
def landscape(t):
    t = t.replace(r"\begin{table}[!htbp]\centering", r"\begin{sidewaystable}\centering\small", 1).replace(r"\end{table}", r"\end{sidewaystable}")
    t = t.replace(r"\begin{tabular}", r"\resizebox{\textheight}{!}{\begin{tabular}", 1).replace(r"\bottomrule\end{tabular}", r"\bottomrule\end{tabular}}", 1)
    return t
sh = np.array(R["det_superellipse"]["GDINO-B (semi)"]["shapes"]); B = dict(P.BASE)

# ------------------------------------------------ Table: nine policies at baseline
order = [("No QC", R2["policies"]), ("F1-optimal", R2["policies"]), ("High-precision (p>=0.90)", R2["policies"]),
         ("Uniform recall 0.80", R2["policies"]), ("Cost-sensitive (grade-blind)", R3["benchmarks"]),
         ("Purity-constrained recovery max", R3["benchmarks"]), ("Global threshold (tuned)", R2["policies"]),
         ("Value-optimal", R2["policies"]), ("Value-optimal, recovery floor", R2["policies"])]
NAME = {"Purity-constrained recovery max": "Purity-constrained recovery", "High-precision (p>=0.90)": r"High-precision ($p\ge$0.90)",
        "Value-optimal": "VO", "Value-optimal, recovery floor": "VO with recovery floor (VO-RF)"}
GROUP = {"No QC": "", "F1-optimal": "a", "High-precision (p>=0.90)": "a", "Uniform recall 0.80": "a",
         "Cost-sensitive (grade-blind)": "b", "Purity-constrained recovery max": "b",
         "Global threshold (tuned)": "c", "Value-optimal": "c", "Value-optimal, recovery floor": "c"}
rows = []; POL = {}
for n, src in order:
    d = src[n]; o = M.evaluate(np.array(d["r"]), sh, B, True); POL[n] = o
    assert abs(float(o["nsv"]) - d["nsv"]) < 1e-9, n          # stored and recomputed values agree
# film grade label computed directly from purity; portrait layout, single-spaced
SHORT = {"No QC": "No QC", "F1-optimal": "F1-optimal", "High-precision (p>=0.90)": "High-precision",
         "Uniform recall 0.80": "Uniform 0.80", "Cost-sensitive (grade-blind)": "Cost-sensitive",
         "Purity-constrained recovery max": "Purity-constrained", "Global threshold (tuned)": "GL",
         "Value-optimal": "VO", "Value-optimal, recovery floor": "VO-RF"}
rows = []; prev = None
for n, src in order:
    o = POL[n]; pur = float(o["film_purity"]); has = float(o["bin_total"][1]) > 1e-9
    fg = ("--" if not has else (f"{f(100*pur,0)} (" + ("A" if pur >= 0.95 else "B" if pur >= 0.90 else "C") + ")"))
    if GROUP[n] and GROUP[n] != GROUP.get(prev, None) and prev is not None: rows.append(r"\addlinespace[3pt]")
    prev = n
    rows.append(f"{SHORT[n]}" + (f"$^{{{GROUP[n]}}}$" if GROUP[n] else "") + f" & {f(float(o['nsv']),2)} & {f(float(o['line_total']),0)} & {G[int(o['grade_line'])]} & {fg} & "
                f"{f(float(o['recovered_total']))} & {f(float(o['misdirected']))} & {f(float(o['paper_lost']))} & {f(float(o['resid_cb']))} & {f(float(o['resid_nonfibre']))} & {f(float(o['picks']),0)}" + r"\\")
tab = "\n".join([
    r"\begin{table}[!htbp]\centering",
    r"\begingroup\linespread{1}\selectfont\footnotesize\setlength{\tabcolsep}{2.5pt}",
    r"\caption{Outcomes of nine operating policies per tonne of paper-line feed (GDINO-B (semi), baseline). Masses in kg per tonne of feed; film grade in parentheses.}\label{tab:policies}",
    r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrlcrrrrrr@{}}\toprule",
    r"Policy & \shortstack{NSV\\(USD\\t$^{-1}$)} & \shortstack{Paper\\prod-\\uct} & \shortstack{Paper\\grade} & \shortstack{Film\\purity\\(\%)} & \shortstack{Re-\\cov-\\ered} & \shortstack{Mis-\\direc-\\ted} & \shortstack{Paper\\lost} & \shortstack{Resid.\\card-\\board} & \shortstack{Resid.\\plast.,\\metal} & \shortstack{Picks\\(t$^{-1}$)}\\\midrule"]
    + rows + [r"\bottomrule\end{tabular*}",
    r"\par\smallskip{\raggedright\footnotesize Groups: $^{a}$specification-agnostic detection rules; $^{b}$benchmarks from cost-sensitive classification and separation design (Section~\ref{sec:benchmarks}); $^{c}$grade-aware policies that optimise the specification-based value $\Pi$ (Section~\ref{sec:model}). GL: global threshold (tuned); VO: value-optimal; VO-RF: value-optimal with recovery floor. NSV: net sales value. SRPN: sorted residential papers; MP: mixed paper. Recovered: target mass in its own bin; misdirected: target objects in a wrong bin; paper lost: paper picked by false positives; residual: contaminants left in the paper product.\par}",
    r"\endgroup", r"\end{table}"])
open(T + "tab_policies_caie.tex", "w").write(tab)

# ------------------------------------------------ Table: computational performance
tm = R3["timing"]; dec = R3["decomposition"]
rows = []
for k in ("14", "20", "30", "50"):
    d = tm[k]
    rows.append(f"Exhaustive enumeration, $k={k}$ & {d['points']:,} & {f(d['t_unc'],2)} & {f(d['v_unc'],3)} & {f(d['t_rf'],2)} & {f(d['v_rf'],3)} & {d['feasible_rf']:,}" + r"\\")
nfu = sum(1 for p in dec["unc"]["patterns"] if p["feasible_starts"] > 0); nfr = sum(1 for p in dec["rf"]["patterns"] if p["feasible_starts"] > 0)
rows.append(r"\addlinespace[2pt]" + f"Grade-pattern decomposition + SLSQP & -- & {f(dec['unc']['t'],2)} & {f(dec['unc']['v'],3)} & {f(dec['rf']['t'],2)} & {f(dec['rf']['v'],3)} & --" + r"\\")
open(T + "tab_comp.tex", "w").write(wrap(
    r"Computational performance and solution quality for problem (P) without and with the recovery floor (GDINO-B (semi), baseline). Times are wall-clock seconds on one core of a 2-vCPU cloud machine (Python 3, NumPy, SciPy). The decomposition solves each grade-pattern subproblem from six grid starts and reports the best solution whose true value $\Pi$ is confirmed by re-evaluation.",
    "tab:comp", "lrrrrrr",
    r"Method & Operating points & \multicolumn{2}{c}{Without recovery floor} & \multicolumn{3}{c}{With recovery floor (VO-RF)}\\\cmidrule(lr){3-4}\cmidrule(lr){5-7}" + "\n" +
    r" & evaluated & Time (s) & $\Pi^\ast$ (USD t$^{-1}$) & Time (s) & $\Pi^\ast$ (USD t$^{-1}$) & Feasible points", rows))

# ------------------------------------------------ Table: grade-pattern subproblems (supplement)
rows = []
for lab, key in (("Without recovery floor", "unc"), ("With recovery floor", "rf")):
    pats = [p for p in dec[key]["patterns"] if p["feasible_starts"] > 0]
    pats = sorted(pats, key=lambda p: -p["v"])
    for p in pats:
        g = p["g"]; rows.append(f"{lab} & {G[g[0]]} & {GO[g[1]]} & {GF[g[2]]} & {p['feasible_starts']:,} & {f(p['v'],3)}" + r"\\")
    lab_inf = [p for p in dec[key]["patterns"] if p["feasible_starts"] == 0]
    rows.append(f"{lab} & \\multicolumn{{3}}{{l}}{{{len(lab_inf)} further patterns}} & 0 & infeasible on grid" + r"\\" + r"\addlinespace[2pt]")
open(T + "supp/tab_patterns.tex", "w").write(wrap(
    r"Grade-pattern subproblems of the exact decomposition (Proposition~1 of the main text), sorted by optimal value. A pattern fixes the grade of the paper product, the cardboard (OCC) bin and the film bin. Feasible starts: grid points ($k=14$) that satisfy the pattern's specification constraints. Value: best confirmed value $\Pi$ of the subproblem (USD t$^{-1}$).",
    "tab:patterns", "llllrr", r"Problem & Paper & Cardboard bin & Film & Feasible starts & Value", rows, resize=True))

# ------------------------------------------------ Table: paired Monte Carlo benchmark
mc = R3["mc"]; PN = [("F1", "F1-optimal"), ("CS", "Cost-sensitive (grade-blind)"), ("PC", "Purity-constrained recovery"),
                     ("GL", "Global threshold (tuned)"), ("RF", "VO with recovery floor (VO-RF)")]
rows = []; MCM = {}
for k, nm in PN:
    g = np.array(mc["gain_" + k]); s = np.array(mc["srpn_" + k]); r = np.array(mc["rec_" + k]); pk = np.array(mc["picks_" + k])
    MCM[k] = dict(mean=g.mean(), med=np.median(g), p5=np.percentile(g, 5), p95=np.percentile(g, 95), srpn=100 * s.mean(), neg=100 * (g < 0).mean(), rec=np.median(r), picks=np.median(pk))
    rows.append(f"{nm} & {f(g.mean(),2)} & {f(np.median(g),2)} & [{f(np.percentile(g,5),2)}, {f(np.percentile(g,95),2)}] & {f(100*s.mean(),1)} & {f(100*(g<0).mean(),1)} & {f(np.median(r))} & {f(np.median(pk),0)}" + r"\\")
open(T + "tab_mcbench.tex", "w").write(wrap(
    r"Paired Monte Carlo comparison of five policies on identical draws (" + f"{len(mc['gain_F1']):,}" + r" draws; all uncertain inputs sampled jointly as in Section~\ref{sec:design-unc}; GDINO-B (semi)). Value of QC: NSV gain over no robotic QC.",
    "tab:mcbench", "lrrcrrrr",
    r"Policy & \multicolumn{3}{c}{Value of QC (USD t$^{-1}$)} & SRPN kept & Value $<0$ & Recovered & Picks\\\cmidrule(lr){2-4}" + "\n" +
    r" & Mean & Median & 5th--95th pct. & (\% draws) & (\% draws) & (kg t$^{-1}$, median) & (t$^{-1}$, median)", rows))

# ------------------------------------------------ macros
gRF = np.array(mc["gain_RF"]); fb = np.array([x for x in mc["f1_below_cliff"] if x is not None]); sF = np.array(mc["srpn_F1"])
agree = float(np.mean(fb == (1 - sF[[i for i, x in enumerate(mc["f1_below_cliff"]) if x is not None]])))
cl = R3["cliff_f1ctx"]; cs = POL["Cost-sensitive (grade-blind)"]; pc = POL["Purity-constrained recovery max"]
mac = dict(
    TenumFifty=f(tm["50"]["t_unc"], 1), TenumFiftyRF=f(tm["50"]["t_rf"], 1), TenumFourteen=f(tm["14"]["t_unc"], 2),
    NptsFifty=f"{tm['50']['points']:,}", NptsFourteen=f"{tm['14']['points']:,}",
    VgridFifty=f(tm["50"]["v_unc"], 3), VgridFiftyRF=f(tm["50"]["v_rf"], 3), VgridFourteen=f(tm["14"]["v_unc"], 3), VgridFourteenRF=f(tm["14"]["v_rf"], 3),
    VgridThirty=f(tm["30"]["v_unc"], 3),
    VdecU=f(dec["unc"]["v"], 3), VdecRF=f(dec["rf"]["v"], 3), TdecU=f(dec["unc"]["t"], 1), TdecRF=f(dec["rf"]["t"], 1),
    GapU=f(dec["unc"]["v"] - tm["50"]["v_unc"], 3), GapRF=f(dec["rf"]["v"] - tm["50"]["v_rf"], 3),
    GapPctU=f(100 * (dec["unc"]["v"] - tm["50"]["v_unc"]) / dec["unc"]["v"], 2),
    NpatFeasU=str(nfu), NpatFeasRF=str(nfr),
    PatU=f"({G[dec['unc']['g'][0]]}, {GO[dec['unc']['g'][1]]}, film {GF[dec['unc']['g'][2]]})",
    PatRF=f"({G[dec['rf']['g'][0]]}, {GO[dec['rf']['g'][1]]}, film {GF[dec['rf']['g'][2]]})",
    Rhat=f(cl["r_hat"], 3), JumpModel=f(cl["jump"], 3), JumpPred=f(cl["jump_pred"], 3), LineAtCliff=f(cl["line_total"], 1),
    ProhibAtCliff=f(100 * cl["prohib_at"], 2), RFoneCB=f(float(R2["policies"]["F1-optimal"]["r"][0]), 3),
    CSnsv=f(float(cs["nsv"]), 2), CSrec=f(float(cs["recovered_total"])), CSfilm=f(100 * float(cs["film_purity"]), 0), CSpicks=f(float(cs["picks"]), 0),
    CSgrade=G[int(cs["grade_line"])], CSmis=f(float(cs["misdirected"])),
    PCnsv=f(float(pc["nsv"]), 2), PCrec=f(float(pc["recovered_total"])), PCgrade=G[int(pc["grade_line"])], PCfilm=f(100 * float(pc["film_purity"]), 0),
    PCout=f(100 * float(pc["outthrow"]), 1), PCpicks=f(float(pc["picks"]), 0), PCloss=f(float(pc["nsv"]) - float(POL["No QC"]["nsv"]), 2),
    NMCthree=f"{len(mc['gain_F1']):,}", TdrawMC=f(float(np.mean(mc["t_draw"])), 2), CliffAgree=f(100 * agree, 0),
    FbelowShare=f(100 * fb.mean(), 1),
    **{f"MC{k}{s}": f(v[s2], d_) for k, v in MCM.items() for s, s2, d_ in (("mean", "mean", 2), ("med", "med", 2), ("lo", "p5", 2), ("hi", "p95", 2), ("srpn", "srpn", 1), ("neg", "neg", 1), ("rec", "rec", 1), ("picks", "picks", 0))},
    RFminusGL=f(float(np.mean(gRF - np.array(mc["gain_GL"]))), 2), RFgeGL=f(100 * float(np.mean(gRF - np.array(mc["gain_GL"]) >= -1e-9)), 0),
    RFminusFone=f(float(np.mean(gRF - np.array(mc["gain_F1"]))), 2), RFgeFone=f(100 * float(np.mean(gRF - np.array(mc["gain_F1"]) >= -1e-9)), 0),
    RFminusCS=f(float(np.mean(gRF - np.array(mc["gain_CS"]))), 2), RFminusPC=f(float(np.mean(gRF - np.array(mc["gain_PC"]))), 2),
    BcsCB=f(1000 * R3["cs_benefit_per_pick"][0], 2), BcsSP=f(1000 * R3["cs_benefit_per_pick"][1], 2),
)
tmax = max(max(dec["unc"]["v"] - tm[k]["v_unc"], dec["rf"]["v"] - tm[k]["v_rf"]) for k in tm)
mac.update(MaxGridGap=f(tmax, 2), CSrSP=f(float(R3["benchmarks"]["Cost-sensitive (grade-blind)"]["r"][1]), 2),
           RFrSP=f(float(R2["policies"]["Value-optimal, recovery floor"]["r"][1]), 2),
           FOnerSP=f(float(R2["policies"]["F1-optimal"]["r"][1]), 2))
mac.update(GLminusRF=f(float(np.mean(np.array(mc["gain_GL"]) - gRF)), 2))
fix = []
for fn in ("../tables/macros.tex", "../tables/macros2.tex"):
    for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(fn).read()):
        if v.startswith("-"): fix.append(r"\renewcommand{\%s}{$-$%s}" % (k, v[1:]))
open(T + "macros_fix.tex", "w").write("\n".join(fix) + "\n")
mac = {k.replace("1", "One").replace("5", "Five"): v for k, v in mac.items()}
for k in mac: assert k.isalpha(), k
open(T + "macros3.tex", "w").write("\n".join(r"\newcommand{\%s}{%s}" % (k, v) for k, v in mac.items()) + "\n")
print(len(mac), "macros"); print({k: mac[k] for k in list(mac)[:80]})
