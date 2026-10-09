"""Revision analyses: physical indicators, baseline policies, recovery-constrained optimum, Pareto front,
scenarios, pick-cost break-even, 2000-draw Monte Carlo with convergence, class-weight intervals."""
import json, time, numpy as np
from scipy.stats import kendalltau
import params as P, model as M
rng = np.random.default_rng(20260928); t0 = time.time()
FOCAL = "GDINO-B (semi)"; BASEP = dict(P.BASE); OUT = {}
KEYS = ["nsv","grade_line","recovered","recovered_total","misdirected","paper_lost","resid_nonfibre","resid_cb",
        "contam_avoided","recovery_eff","spec_mass","downgraded","picks","paper_purity","purity","recovery","prohib","outthrow"]
def pack(o): return {k: (np.asarray(o[k]).tolist()) for k in KEYS}

def optimize_c(shapes, prm, k=50, rmax=0.98, cons=None):
    g = np.linspace(0, rmax, k); A, B, C = np.meshgrid(g, g, g, indexing="ij")
    rest = np.stack([A.ravel(), B.ravel(), C.ravel()], -1); best_v, best_r = -np.inf, None
    for r0 in g:
        Rg = np.concatenate([np.full((rest.shape[0], 1), r0), rest], 1)
        if cons is None: v = M.evaluate(Rg, shapes, prm)
        else:
            o = M.evaluate(Rg, shapes, prm, True); v = np.where(cons(o), o["nsv"], -np.inf)
        i = int(np.argmax(v))
        if v[i] > best_v: best_v, best_r = float(v[i]), Rg[i].copy()
    return best_r, best_v

def policies(shapes, prm, k=50):
    fam = prm["pr_family"]; pol = {}
    pol["No QC"] = np.zeros(4)
    pol["F1-optimal"] = M.budget_scaled_f1(shapes, prm)
    pol["High-precision (p>=0.90)"] = M.hp_policy(shapes, fam)
    pol["Uniform recall 0.80"] = np.full(4, 0.80)
    pol["Global threshold (tuned)"] = M.global_opt(shapes, prm)[0]
    pol["Value-optimal"] = optimize_c(shapes, prm, k)[0]
    eff_f1 = float(M.evaluate(pol["F1-optimal"], shapes, prm, True)["recovery_eff"])
    pol["Value-optimal, recovery floor"] = optimize_c(shapes, prm, k,
        cons=lambda o: (o["recovery_eff"] >= eff_f1 - 1e-9) & (o["p_film"] > 0) & (o["downgraded"] <= 1000))[0]
    return pol, eff_f1

sh = M.shapes_for(FOCAL, "superellipse")
pol, eff_f1 = policies(sh, BASEP)
OUT["policies"] = {n: dict(r=r.tolist(), **pack(M.evaluate(r, sh, BASEP, True))) for n, r in pol.items()}
OUT["eff_f1"] = eff_f1
print("policies", time.time() - t0)

# ---------------- Pareto front NSV vs recovered mass (k=30 grid)
g = np.linspace(0, 0.98, 30); A_, B_, C_, D_ = np.meshgrid(g, g, g, g, indexing="ij")
Rall = np.stack([A_.ravel(), B_.ravel(), C_.ravel(), D_.ravel()], -1)
nsv_all, rec_all, spec_all = [], [], []
for ch in np.array_split(Rall, 20):
    o = M.evaluate(ch, sh, BASEP, True); nsv_all.append(o["nsv"]); rec_all.append(o["recovered_total"]); spec_all.append(o["spec_mass"])
nsv_all = np.concatenate(nsv_all); rec_all = np.concatenate(rec_all)
bins = np.linspace(0, rec_all.max(), 60); front = []
for i in range(len(bins) - 1):
    msk = rec_all >= bins[i]
    if msk.any(): j = np.argmax(np.where(msk, nsv_all, -np.inf)); front.append((float(bins[i]), float(nsv_all[j]), float(rec_all[j])))
OUT["pareto"] = front
idx = np.random.default_rng(1).choice(len(nsv_all), 6000, replace=False)
OUT["cloud"] = dict(nsv=nsv_all[idx].tolist(), rec=rec_all[idx].tolist())
print("pareto", time.time() - t0)

# ---------------- Scenarios: contamination x price x specification
STRICT = dict(P.GRADES, srpn_prohib=0.01, srpn_outthrow=0.02, mp_prohib=0.015)
sc = {}
for wlab, w in (("Low (4%)", 0.04), ("Medium (8%)", 0.08), ("High (14%)", 0.14)):
    for plab in ("2026-06", "2025-06"):
        for slab, G in (("Standard", P.GRADES), ("Strict", STRICT)):
            prm = dict(BASEP, w=w, price=plab, grades=G)
            pl, ef = policies(sh, prm, k=30)
            sc[f"{wlab}|{plab}|{slab}"] = {n: dict(nsv=float(M.evaluate(r, sh, prm)), **{kk: float(np.asarray(M.evaluate(r, sh, prm, True)[kk])) for kk in ("recovered_total","spec_mass","grade_line","picks","recovery_eff")}) for n, r in pl.items()}
OUT["scenarios"] = sc
print("scenarios", time.time() - t0)

# ---------------- Pick cost: break-even and cost-aware optimum
pc = {}
none = OUT["policies"]["No QC"]["nsv"]
for n in ("F1-optimal", "Value-optimal", "Value-optimal, recovery floor", "Global threshold (tuned)"):
    d = OUT["policies"][n]; pc[n] = dict(break_even=(d["nsv"] - none) / max(d["picks"], 1e-9))
for c in (0.0, 0.002, 0.005, 0.01, 0.02):
    prm = dict(BASEP, c_pick=c); r, v = optimize_c(sh, prm, 30)
    o = M.evaluate(r, sh, prm, True); f1 = float(M.evaluate(M.f1_policy(sh, "superellipse"), sh, prm))
    pc[f"c={c}"] = dict(nsv_net=v, nsv_f1_net=f1, picks=float(o["picks"]), recovered=float(o["recovered_total"]), grade=int(o["grade_line"]), r=r.tolist())
OUT["pick_cost"] = pc
print("pick cost", time.time() - t0)

# ---------------- Monte Carlo (2000 draws)
NMC = 2000; names = list(P.DETECTORS); mAP = np.array([P.DETECTORS[d][1] for d in names])
mc = {k: [] for k in ("g_f1","g_opt","g_rf","g_opt_f1","srpn_f1","srpn_opt","srpn_rf","r_opt","r_rf","rec_f1","rec_opt","rec_rf",
                      "spec_f1","spec_opt","spec_rf","pp_opt","pp_f1","resid_f1","resid_opt","resid_rf","tau","top")}
def draw():
    prm = dict(BASEP)
    prm["w"] = rng.uniform(*P.RANGES["w"]); prm["m"] = BASEP["m"] * np.exp(rng.normal(0, 0.35, 4))
    prm["m_paper"] = rng.uniform(*P.RANGES["m_paper"]); prm["phi"] = rng.uniform(*P.RANGES["phi"])
    prm["eta"] = rng.uniform(*P.RANGES["eta"]); prm["al_share"] = rng.uniform(*P.RANGES["al_share"])
    prm["disposal"] = rng.uniform(*P.RANGES["disposal"]); prm["price"] = "2025-06" if rng.random() < 0.5 else "2026-06"
    prm["pr_family"] = "power" if rng.random() < 0.5 else "superellipse"; return prm
for i in range(NMC):
    prm = draw(); shd = M.shapes_for(FOCAL, prm["pr_family"])
    nn = float(M.evaluate(np.zeros(4), shd, prm))
    rf1 = M.f1_policy(shd, prm["pr_family"]); o1 = M.evaluate(rf1, shd, prm, True)
    ro, vo = optimize_c(shd, prm, 14); oo = M.evaluate(ro, shd, prm, True)
    ef = float(o1["recovery_eff"])
    rr, vr = optimize_c(shd, prm, 14, cons=lambda o: (o["recovery_eff"] >= ef * 0.999) & (o["p_film"] > 0))
    if not np.isfinite(vr): rr = rf1
    orr = M.evaluate(rr, shd, prm, True)
    for key, o in (("f1", o1), ("opt", oo), ("rf", orr)):
        mc["rec_" + key].append(float(o["recovered_total"])); mc["spec_" + key].append(float(o["spec_mass"]))
        mc["srpn_" + key].append(int(o["grade_line"] == 2)); mc["resid_" + key].append(float(o["resid_nonfibre"] + o["resid_cb"]))
    mc["g_f1"].append(float(o1["nsv"]) - nn); mc["g_opt"].append(float(oo["nsv"]) - nn); mc["g_rf"].append(float(orr["nsv"]) - nn)
    mc["g_opt_f1"].append(float(oo["nsv"] - o1["nsv"])); mc["r_opt"].append(ro.tolist()); mc["r_rf"].append(np.asarray(rr).tolist())
    mc["pp_opt"].append(float(oo["paper_purity"])); mc["pp_f1"].append(float(o1["paper_purity"]))
    if i % 2 == 0:   # detector ranking on every second draw (1000 draws) for tractability
        v = np.array([optimize_c(M.shapes_for(d, prm["pr_family"]), prm, 11)[1] for d in names])
        mc["tau"].append(float(kendalltau(mAP, v).statistic)); mc["top"].append(int(np.argmax(v) in np.where(mAP == mAP.max())[0]))
    if i % 250 == 0: print("mc", i, time.time() - t0, flush=True)
OUT["mc"] = mc
# ---------------- class-weight intervals (300 draws)
cw = []
for i in range(300):
    prm = draw(); ap0 = np.array(M.class_ap50(FOCAL))
    def v_of(ap): return optimize_c(np.array([M.calibrate(a, prm["pr_family"]) for a in ap]), prm, 12)[1]
    b = v_of(ap0); dv = []
    for c in range(4):
        a = ap0.copy(); a[c] = min(0.98, a[c] + 0.05); dv.append(v_of(a) - b)
    dv = np.maximum(np.array(dv), 0); cw.append((dv / dv.sum()).tolist() if dv.sum() > 1e-9 else [np.nan] * 4)
OUT["cw_mc"] = cw
json.dump(OUT, open("../results/results_revision.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
print("saved", time.time() - t0)
