"""Runs every analysis reported in the manuscript and writes results.json, LaTeX macros and tables."""
import json, time, numpy as np
from scipy.stats import kendalltau, spearmanr
import params as P, model as M

rng = np.random.default_rng(20260927)
OUT = {}
K = 50
BASEP = dict(P.BASE)
FOCAL = "GDINO-B (semi)"

def run_detector(name, prm, k=K):
    sh = M.shapes_for(name, prm["pr_family"])
    none = float(M.evaluate(np.zeros(4), sh, prm))
    r1 = M.budget_scaled_f1(sh, prm)
    f1 = M.evaluate(r1, sh, prm, True)
    ro, vo = M.optimize(sh, prm, k=k)
    opt = M.evaluate(ro, sh, prm, True)
    return dict(shapes=sh.tolist(), ap50=M.class_ap50(name).tolist(), none=none,
                r_f1=r1.tolist(), nsv_f1=float(f1["nsv"]), r_opt=ro.tolist(), nsv_opt=float(opt["nsv"]),
                grade_f1=int(f1["grade_line"]), grade_opt=int(opt["grade_line"]),
                film_pur_f1=float(f1["film_purity"]), film_pur_opt=float(opt["film_purity"]),
                outthrow_f1=float(f1["outthrow"]), outthrow_opt=float(opt["outthrow"]),
                prohib_f1=float(f1["prohib"]), prohib_opt=float(opt["prohib"]),
                picks_f1=float(f1["picks"]), picks_opt=float(opt["picks"]),
                dec_f1=[float(f1[k_]) for k_ in ("rev_line","rev_occ","rev_film","rev_rigid","rev_metal")],
                dec_opt=[float(opt[k_]) for k_ in ("rev_line","rev_occ","rev_film","rev_rigid","rev_metal")],
                pur_opt=opt["purity"].tolist(), rec_opt=opt["recovery"].tolist(),
                pur_f1=f1["purity"].tolist(), rec_f1=f1["recovery"].tolist(),
                occ_pro_opt=float(opt["occ_pro"]), occ_pro_f1=float(f1["occ_pro"]),
                n=opt["n"].tolist())

t0 = time.time()
# ---------------------------------------------------------------- 1. detector comparison (two PR families)
for fam in ("superellipse", "power"):
    prm = dict(BASEP, pr_family=fam)
    res = {d: run_detector(d, prm) for d in P.DETECTORS}
    names = list(P.DETECTORS)
    mAP = np.array([P.DETECTORS[d][1] for d in names]); mAP50 = np.array([P.DETECTORS[d][2] for d in names])
    nsv = np.array([res[d]["nsv_opt"] for d in names]); nsvf = np.array([res[d]["nsv_f1"] for d in names])
    OUT[f"det_{fam}"] = res
    OUT[f"rank_{fam}"] = dict(
        tau_map_opt=float(kendalltau(mAP, nsv).statistic), tau_map50_opt=float(kendalltau(mAP50, nsv).statistic),
        tau_map_f1=float(kendalltau(mAP, nsvf).statistic), rho_map_opt=float(spearmanr(mAP, nsv).statistic))
print("detectors done", time.time() - t0)

# ---------------------------------------------------------------- 2. pick-budget scenarios (focal detector)
bud = {}
for B in (np.inf, 1800, 1400, 1000, 600, 300):
    prm = dict(BASEP, budget=B)
    r = run_detector(FOCAL, prm)
    bud["inf" if np.isinf(B) else str(B)] = r
OUT["budget"] = bud
print("budget done", time.time() - t0)

# ---------------------------------------------------------------- 3. economic class weights (finite differences)
def nsv_with_ap50(ap50, prm, k=30):
    sh = np.array([M.calibrate(a, prm["pr_family"]) for a in ap50])
    return M.optimize(sh, prm, k=k)[1]
base_ap = np.array(M.class_ap50(FOCAL))
v0 = nsv_with_ap50(base_ap, BASEP)
dv = []
for c in range(4):
    a = base_ap.copy(); a[c] = min(0.98, a[c] + 0.05)
    dv.append(nsv_with_ap50(a, BASEP) - v0)
dv = np.array(dv)
OUT["class_weights"] = dict(v0=v0, dv=dv.tolist(), share=(np.maximum(dv, 0) / max(np.maximum(dv, 0).sum(), 1e-9)).tolist())
print("weights done", time.time() - t0)

# ---------------------------------------------------------------- 4. sweeps for frontier figures (focal)
sh = M.shapes_for(FOCAL, "superellipse")
ro = np.array(OUT["det_superellipse"][FOCAL]["r_opt"]); r1 = np.array(OUT["det_superellipse"][FOCAL]["r_f1"])
grid = np.linspace(0, 0.98, 197)
sweeps = {}
for c in range(4):
    Rg = np.tile(ro, (grid.size, 1)); Rg[:, c] = grid
    o = M.evaluate(Rg, sh, BASEP, True)
    sweeps[c] = dict(r=grid.tolist(), nsv=o["nsv"].tolist(), outthrow=o["outthrow"].tolist(),
                     prohib=o["prohib"].tolist(), purity=o["purity"][:, c].tolist(),
                     recovery=o["recovery"][:, c].tolist(), film_purity=o["film_purity"].tolist(),
                     occ_pro=o["occ_pro"].tolist(), grade=o["grade_line"].tolist())
OUT["sweeps"] = sweeps
# film-bin frontier holding all other classes at the F1 point (conventional setting)
Rg = np.tile(r1, (grid.size, 1)); Rg[:, 1] = grid
o = M.evaluate(Rg, sh, BASEP, True)
OUT["film_frontier_f1ctx"] = dict(r=grid.tolist(), purity=o["purity"][:, 1].tolist(), recovery=o["recovery"][:, 1].tolist(),
                                   nsv=o["nsv"].tolist(), rev_film=o["rev_film"].tolist())
# near-optimal plateau size
g = np.linspace(0, 0.98, 30)
A_, B_, C_, D_ = np.meshgrid(g, g, g, g, indexing="ij")
Rall = np.stack([A_.ravel(), B_.ravel(), C_.ravel(), D_.ravel()], -1)
vals = M.evaluate(Rall, sh, BASEP)
best = vals.max()
OUT["plateau"] = dict(best=float(best), share_within_0p5=float(np.mean(vals >= best - 0.5)),
                      share_within_1=float(np.mean(vals >= best - 1.0)),
                      share_srpn=float(np.mean(M.evaluate(Rall, sh, BASEP, True)["grade_line"] == 2)))
print("sweeps done", time.time() - t0)

# ---------------------------------------------------------------- 5. one-at-a-time sensitivity (focal)
def gain_pair(prm, k=30):
    sh = M.shapes_for(FOCAL, prm["pr_family"])
    none = float(M.evaluate(np.zeros(4), sh, prm))
    f1 = float(M.evaluate(M.budget_scaled_f1(sh, prm), sh, prm))
    opt = M.optimize(sh, prm, k=k)[1]
    return none, f1, opt
base_trip = gain_pair(BASEP)
tor = {}
def setp(key, val):
    prm = dict(BASEP)
    if key == "m_scale": prm["m"] = BASEP["m"] * val
    elif key == "price": prm["price"] = val
    elif key == "pr_family": prm["pr_family"] = val
    else: prm[key] = val
    return prm
for key, (lo, hi) in P.RANGES.items():
    tor[key] = [gain_pair(setp(key, lo)), gain_pair(setp(key, hi))]
tor["price"] = [gain_pair(setp("price", "2025-06")), gain_pair(setp("price", "2026-06"))]
tor["pr_family"] = [gain_pair(setp("pr_family", "power")), gain_pair(setp("pr_family", "superellipse"))]
OUT["tornado"] = dict(base=base_trip, runs=tor)
print("tornado done", time.time() - t0)

# ---------------------------------------------------------------- 6. Monte Carlo
NMC = 500; KMC = 14
names = list(P.DETECTORS); mAP = np.array([P.DETECTORS[d][1] for d in names])
mc = dict(gain_opt_none=[], gain_opt_f1=[], gain_f1_none=[], srpn_f1=[], srpn_opt=[], tau=[], top_is_map_top=[], best_det=[])
for i in range(NMC):
    prm = dict(BASEP)
    prm["w"] = rng.uniform(*P.RANGES["w"])
    prm["m"] = BASEP["m"] * np.exp(rng.normal(0, 0.35, 4))
    prm["m_paper"] = rng.uniform(*P.RANGES["m_paper"])
    prm["phi"] = rng.uniform(*P.RANGES["phi"])
    prm["eta"] = rng.uniform(*P.RANGES["eta"])
    prm["al_share"] = rng.uniform(*P.RANGES["al_share"])
    prm["disposal"] = rng.uniform(*P.RANGES["disposal"])
    prm["price"] = "2025-06" if rng.random() < 0.5 else "2026-06"
    prm["pr_family"] = "power" if rng.random() < 0.5 else "superellipse"
    nsv_det = []
    for d in names:
        shd = M.shapes_for(d, prm["pr_family"])
        v = M.optimize(shd, prm, k=KMC)[1]
        nsv_det.append(v)
        if d == FOCAL:
            none = float(M.evaluate(np.zeros(4), shd, prm))
            o1 = M.evaluate(M.f1_policy(shd, prm["pr_family"]), shd, prm, True)
            ro_, _ = M.optimize(shd, prm, k=KMC); oo = M.evaluate(ro_, shd, prm, True)
            mc["gain_opt_none"].append(v - none); mc["gain_opt_f1"].append(v - float(o1["nsv"]))
            mc["gain_f1_none"].append(float(o1["nsv"]) - none)
            mc["srpn_f1"].append(int(o1["grade_line"] == 2)); mc["srpn_opt"].append(int(oo["grade_line"] == 2))
    nsv_det = np.array(nsv_det)
    mc["tau"].append(float(kendalltau(mAP, nsv_det).statistic))
    mc["best_det"].append(names[int(np.argmax(nsv_det))])
    mc["top_is_map_top"].append(int(np.argmax(nsv_det) in np.where(mAP == mAP.max())[0]))
OUT["mc"] = mc
print("mc done", time.time() - t0)

class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)
json.dump(OUT, open("../results/results.json", "w"), cls=NpEnc, indent=1)
print("saved", time.time() - t0)
