import json, numpy as np, params as P, model as M
R2 = json.load(open("../results/results_revision.json")); T = "../tables/"; mc = R2["mc"]
G = {2: "SRPN", 1: "MP", 0: "Off-spec"}; f = lambda x, d=1: f"{x:.{d}f}"
def wrap(cap, lab, colspec, head, rows):
    return "\n".join([r"\begin{table}[!htbp]\centering\caption{" + cap + r"}\label{" + lab + "}", r"\resizebox{\textwidth}{!}{\begin{tabular}{" + colspec + r"}\toprule", head + r"\\\midrule"] + rows + [r"\bottomrule\end{tabular}}\end{table}"])
# ---- policies (physical + value)
rows = []; LINE = {}
_sh = np.array(json.load(open("../results/results.json"))["det_superellipse"]["GDINO-B (semi)"]["shapes"])
for n, d in R2["policies"].items():
    LINE[n] = float(M.evaluate(np.array(d["r"]), _sh, dict(P.BASE), True)["line_total"])
    rec = d["recovered"]
    rows.append(f"{n.replace('>=', '$\\ge$')} & {f(d['nsv'],2)} & {f(LINE[n],0)} {G[d['grade_line']]} & " + " & ".join(f(x) for x in rec) +
                f" & {f(d['recovered_total'])} & {f(d['misdirected'])} & {f(d['paper_lost'])} & {f(d['resid_cb'])} & {f(d['resid_nonfibre'])} & {f(d['spec_mass'],0)} & {f(d['picks'],0)}" + r"\\")
MAC_LINE = dict(LineNoQC=f(LINE["No QC"],0), LineFone=f(LINE["F1-optimal"],0), LineRF=f(LINE["Value-optimal, recovery floor"],0))
open(T + "tab_policies.tex", "w").write(wrap(
 r"Resource-recovery and value outcomes per tonne of paper-line feed for seven operating policies (GDINO-B (semi), baseline). Recovered: target material in its own bin, whatever the grade of that bin; misdirected: target objects in a wrong bin; paper lost: paper picked by false positives; residual: contaminants left in the paper product; at-spec: mass leaving at a positive-value grade. Paper product: mass and the grade at which it sells (SRPN, sorted residential papers; MP, mixed paper).",
 "tab:policies", "lrlrrrrrrrrrrr",
 r"Policy & NSV & Paper & \multicolumn{4}{c}{Recovered (kg t$^{-1}$)} & Total & Misdir. & Paper & \multicolumn{2}{c}{Residual (kg t$^{-1}$)} & At-spec & Picks\\\cmidrule(lr){4-7}\cmidrule(lr){11-12}" + "\n" +
 r" & (USD t$^{-1}$) & (kg t$^{-1}$, grade) & CB & SP & RP & M & (kg t$^{-1}$) & (kg t$^{-1}$) & lost & Cardboard & Plastics+metal & (kg t$^{-1}$) & (t$^{-1}$)", rows))
# ---- scenarios
rows = []
for k, v in R2["scenarios"].items():
    w, pr, sp = k.split("|"); w = w.replace("%", r"\%"); base = v["No QC"]
    rows.append(f"{w} & {pr} & {sp} & {f(base['nsv'],2)} ({G[int(base['grade_line'])]}) & " + " & ".join(f(v[p]['nsv'] - base['nsv'], 2) for p in ("F1-optimal", "High-precision (p>=0.90)", "Global threshold (tuned)", "Value-optimal", "Value-optimal, recovery floor")) + f" & {f(v['Value-optimal, recovery floor']['recovered_total'])}" + r"\\")
open(T + "tab_scenarios.tex", "w").write(wrap(
 r"Transferability scenarios: value of robotic QC relative to no QC (USD t$^{-1}$) for three contamination levels, two price years and two bale-specification environments (strict: prohibitives $\le$1\% and outthrows $\le$2\% for SRPN, prohibitives $\le$1.5\% for mixed paper; an illustrative tighter buyer specification).",
 "tab:scenarios", "lllrrrrrrr",
 r"Target share & Prices & Spec. & No QC NSV & F1 & High-prec. & Global & Value-opt. & VO + floor & Recovered$^{\mathrm{VO+floor}}$ (kg t$^{-1}$)", rows))
# ---- MC summary with intervals
def q(a): a = np.array(a, float); a = a[~np.isnan(a)]; return [np.mean(a), np.percentile(a, 5), np.median(a), np.percentile(a, 95)]
g = lambda k: np.array(mc[k], float)
cw = np.array(R2["cw_mc"], float)
items = [("Value of QC, F1 thresholds (USD t$^{-1}$)", g("g_f1")),
         ("Value of QC, value-optimal (USD t$^{-1}$)", g("g_opt")),
         ("Value of QC, value-optimal with recovery floor (USD t$^{-1}$)", g("g_rf")),
         ("Gain of recovery-floor policy over F1 (USD t$^{-1}$)", g("g_rf") - g("g_f1")),
         ("Target material recovered, F1 (kg t$^{-1}$)", g("rec_f1")),
         ("Target material recovered, recovery-floor policy (kg t$^{-1}$)", g("rec_rf")),
         ("Contaminants left in paper, F1 (kg t$^{-1}$)", g("resid_f1")),
         ("Contaminants left in paper, recovery-floor policy (kg t$^{-1}$)", g("resid_rf")),
         ("Mass sold at specification grade, F1 (kg t$^{-1}$)", g("spec_f1")),
         ("Mass sold at specification grade, recovery-floor (kg t$^{-1}$)", g("spec_rf")),
         ("Paper-product purity, F1 (\\%)", 100*g("pp_f1")),
         ("Paper-product purity, value-optimal (\\%)", 100*g("pp_opt")),
         ("Optimal cardboard recall, value-optimal policy", np.array(mc["r_opt"])[:, 0]),
         ("Optimal cardboard recall, recovery-floor policy", np.array(mc["r_rf"])[:, 0]),
         ("Cardboard share of marginal accuracy value", cw[:, 0]),
         ("Soft-plastic share of marginal accuracy value", cw[:, 1]),
         ("Kendall $\\tau$, mAP vs.\\ value rank (12 detectors)", g("tau"))]
rows = [f"{lab} & " + " & ".join(f(x, 2) for x in q(a)) + r"\\" for lab, a in items]
rows.append(r"\midrule \multicolumn{5}{l}{Probabilities: paper keeps SRPN grade under F1 " + f"{100*np.mean(mc['srpn_f1']):.0f}\\%, value-optimal {100*np.mean(mc['srpn_opt']):.0f}\\%, recovery floor {100*np.mean(mc['srpn_rf']):.0f}\\%; F1 worth less than no QC {100*np.mean(g('g_f1')<0):.0f}\\%; top-mAP detector is top-value {100*np.mean(mc['top']):.0f}\\%" + r".}\\")
open(T + "tab_mc.tex", "w").write(wrap(
 f"Monte Carlo results ({len(mc['g_f1'])} draws; detector ranking on {len(mc['tau'])} draws, class weights on {int((~np.isnan(cw[:,0])).sum())} draws). All assumptions in Table~\\ref{{tab:params}} sampled jointly; price year and PR family drawn at random.",
 "tab:mc", "lrrrr", r"Quantity & Mean & P5 & Median & P95", rows))
# ---- macros
d = R2["policies"]; F1 = d["F1-optimal"]; RF = d["Value-optimal, recovery floor"]; VO = d["Value-optimal"]; GL = d["Global threshold (tuned)"]; HP = d["High-precision (p>=0.90)"]; U8 = d["Uniform recall 0.80"]; NQ = d["No QC"]
pc = R2["pick_cost"]; diff = g("g_rf") - g("g_f1"); fail = 1 - np.array(mc["srpn_f1"])
cond_fail = diff[fail == 1]; cond_ok = diff[fail == 0]
Mc = dict(RFnsv=f(RF["nsv"],2), RFgain=f(RF["nsv"]-NQ["nsv"],2), RFoverF=f(RF["nsv"]-F1["nsv"],2), RFrec=f(RF["recovered_total"]), Frec=f(F1["recovered_total"]),
          VOrec=f(VO["recovered_total"]), RFmis=f(RF["misdirected"]), Fmis=f(F1["misdirected"]), VOmis=f(VO["misdirected"]),
          RFpicks=f(RF["picks"],0), RFfilm=f(100*RF["purity"][1],0), Ffilm=f(100*F1["purity"][1],0), RFcb=f(RF["recovered"][0]), Fcb=f(F1["recovered"][0]),
          RFplost=f(RF["paper_lost"]), Fplost=f(F1["paper_lost"]), Urec=f(U8["recovered_total"]), Uplost=f(U8["paper_lost"]), Unsv=f(U8["nsv"],2), Upicks=f(U8["picks"],0),
          HPnsv=f(HP["nsv"],2), GLnsv=f(GL["nsv"],2), GLrec=f(GL["recovered_total"]), NQresCB=f(NQ["resid_cb"]), NQresNF=f(NQ["resid_nonfibre"]),
          RFresCB=f(RF["resid_cb"]), RFresNF=f(RF["resid_nonfibre"]), RFcontam=f(NQ["resid_cb"]+NQ["resid_nonfibre"]-RF["resid_cb"]-RF["resid_nonfibre"]),
          Fspec=f(F1["spec_mass"],0), RFspec=f(RF["spec_mass"],0), RFeff=f(100*RF["recovery_eff"],0), Feff=f(100*F1["recovery_eff"],0),
          BEf=f(1000*pc["F1-optimal"]["break_even"],1), BErf=f(1000*pc["Value-optimal, recovery floor"]["break_even"],1),
          NMCtwo=str(len(mc["g_f1"])), MCfail=f(100*np.mean(fail),0), MCrfSRPN=f(100*np.mean(mc["srpn_rf"]),0), MCkeepF=f(100*np.mean(mc["srpn_f1"]),0), MCnegF=f(100*np.mean(g("g_f1")<0),0),
          MCrfMean=f(np.mean(diff),2), MCrfLo=f(np.percentile(diff,5),2), MCrfHi=f(np.percentile(diff,95),2), MCcondFail=f(np.mean(cond_fail),2), MCcondOk=f(np.mean(cond_ok),2),
          MCrecLo=f(np.percentile(g("rec_rf"),5)), MCrecHi=f(np.percentile(g("rec_rf"),95)), MCrecMed=f(np.median(g("rec_rf"))),
          MCgainLo=f(np.percentile(g("g_rf"),5),2), MCgainHi=f(np.percentile(g("g_rf"),95),2), MCgainMed=f(np.median(g("g_rf")),2),
          MCcbMed=f(100*np.nanmedian(cw[:,0]),0), MCcbTop=f(100*np.mean(np.argmax(cw[~np.isnan(cw[:,0])],axis=1)==0),0), MCcwN=str(int((~np.isnan(cw[:,0])).sum())),
          MCcbLo=f(100*np.nanpercentile(cw[:,0],25),0), MCcbHi=f(100*np.nanpercentile(cw[:,0],75),0),
          MCtauMed=f(np.median(g("tau")),2), MCtop=f(100*np.mean(mc["top"]),0), MCrCBlo=f(np.percentile(np.array(mc["r_rf"])[:,0],5),2), MCrCBhi=f(np.percentile(np.array(mc["r_rf"])[:,0],95),2),
          ConvHalf=f(np.mean(diff[:len(diff)//2]),2), ConvFull=f(np.mean(diff),2), MCrfWin=f(100*np.mean(diff>0.5),0))
Mc.update(MAC_LINE)
open(T + "macros2.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in Mc.items()) + "\n")
print("\n".join(f"{k}={v}" for k, v in Mc.items()))
R1 = json.load(open("../results/results.json"))
with open(T + "macros2.tex", "a") as fh:
    fh.write("\\newcommand{\\OccProF}{%.1f}\n" % (100 * R1["det_superellipse"]["GDINO-B (semi)"]["occ_pro_f1"]))
# ---- OCC-bin grade and marketable target mass (JEM revision): recovered cardboard can leave in an off-spec bale
def _occ(name):
    o = M.evaluate(np.array(d[name]["r"]), _sh, dict(P.BASE), True); bt, tg = o["bin_total"], o["bin_target"]
    mkt = (tg[0] if o["p_occ"] > 0 else 0.0) + (tg[1] if o["p_film"] > 0 else 0.0) + tg[2] + tg[3]
    return bt[0], bt[0] - tg[0], 100 * (o["occ_pro"] + o["occ_out"]), mkt
_oF, _nF, _sF, _mF = _occ("F1-optimal"); _oR, _nR, _sR, _mR = _occ("Value-optimal, recovery floor"); _oV, _nV, _sV, _mV = _occ("Value-optimal")
with open(T + "macros2.tex", "a") as fh:
    for k, v in dict(OccBinF=f(_oF), OccNonF=f(_nF), OccShareF=f(_sF, 0), OccBinRF=f(_oR), OccNonRF=f(_nR), OccShareRF=f(_sR, 0),
                     OccBinVO=f(_oV), MktRF=f(_mR), MktF=f(_mF), VOspec=f(VO["spec_mass"], 0), NQresAll=f(NQ["resid_cb"] + NQ["resid_nonfibre"])).items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

pc = R2["pick_cost"]; rows = []
for c in ("0.0", "0.002", "0.005", "0.01", "0.02"):
    d = pc["c=" + c]
    rows.append(f"{1000*float(c):.0f} & {d['nsv_f1_net']:.2f} & {d['nsv_net']:.2f} & {d['picks']:.0f} & {d['recovered']:.1f} & {G[d['grade']]}" + r"\\")
open(T + "tab_pickcost.tex", "w").write(wrap(
 r"Sensitivity to a robot cost per pick $c_{\mathrm{pick}}$ (USD per 1,000 picks): net value of F1 thresholds and of the cost-aware value optimum (GDINO-B (semi), baseline, 30-point grid). Break-even costs: " + f"{1000*pc['F1-optimal']['break_even']:.1f}" + r" (F1) and " + f"{1000*pc['Value-optimal, recovery floor']['break_even']:.1f}" + r" (recovery floor) USD per 1,000 picks.",
 "tab:pickcost", "rrrrrl", r"$c_{\mathrm{pick}}$ & Net NSV, F1 & Net NSV, cost-aware opt. & Picks t$^{-1}$ & Recovered (kg t$^{-1}$) & Paper grade", rows))
