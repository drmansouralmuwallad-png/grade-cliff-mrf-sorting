"""CAIE analyses: optimisation-model properties and computational study.

1. Exact grade-pattern decomposition of the threshold problem and continuous refinement
   (bounds the error of the recall grid used by exhaustive enumeration).
2. Computational performance of exhaustive enumeration for grid sizes k = 14, 20, 30, 50.
3. Two additional benchmark policies from the classification and separation literature:
   cost-sensitive (grade-blind) thresholds and purity-constrained recovery maximisation.
4. Numerical checks of Propositions 1-2 (grade cliff location and jump size).
5. Paired Monte Carlo (2000 draws) comparing F1, cost-sensitive, purity-constrained,
   global-threshold and recovery-floor policies on identical draws.
Writes ../results/results_caie.json. Run time about 25-40 min on 2 cores.
"""
import json, time, itertools, numpy as np
from scipy.optimize import minimize, brentq
import params as P, model as M

FOCAL = "GDINO-B (semi)"; BASEP = dict(P.BASE); OUT = {}; t0 = time.time()
RMAX = 0.98
DELTA = 1e-7       # margin kept inside specification limits by the continuous solver
import os; TEST = os.environ.get("CAIE_TEST") == "1"

# ------------------------------------------------------------------ helpers
def price_maps(prm):
    pr = P.PRICES[prm["price"]]
    PL = {2: pr["srpn"], 1: pr["mp"], 0: prm["p_off"]}          # paper product
    PO = {2: pr["occ"], 1: pr["mp"], 0: prm["p_off"]}           # cardboard (OCC) bin
    PF = {2: pr["film_a"], 1: pr["film_b"], 0: pr["film_c"]}    # film bin
    return PL, PO, PF

def pattern_value(o, g, prm):
    """Value with stream prices fixed by grade pattern g = (paper, occ, film)."""
    PL, PO, PF = price_maps(prm)
    v = (o["line_total"] * PL[g[0]] + o["bin_total"][..., 0] * PO[g[1]] + o["bin_total"][..., 1] * PF[g[2]]) / 1000.0
    v = v + o["rev_rigid"] + o["rev_metal"] - prm.get("c_pick", 0.0) * o["picks"]
    return v

def pattern_slacks(o, g, prm):
    """Constraint slacks (>= 0 feasible) defining grade pattern g."""
    G = prm.get("grades", P.GRADES); s = []
    if g[0] == 2: s += [G["srpn_prohib"] - o["prohib"], G["srpn_outthrow"] - o["outthrow"]]
    if g[0] == 1: s += [G["mp_prohib"] - o["prohib"]]
    if g[1] == 2: s += [G["occ_prohib"] - o["occ_pro"], G["occ_total"] - o["occ_pro"] - o["occ_out"]]
    if g[1] == 1: s += [G["mp_prohib"] - o["occ_pro"]]
    if g[2] == 2: s += [o["film_purity"] - G["film_a"]]
    if g[2] == 1: s += [o["film_purity"] - G["film_b"]]
    return s

def enum(shapes, prm, k, objective="nsv", cons=None):
    """Exhaustive enumeration on a k^4 recall grid (exact on the grid)."""
    g = np.linspace(0, RMAX, k); A, B, C = np.meshgrid(g, g, g, indexing="ij")
    rest = np.stack([A.ravel(), B.ravel(), C.ravel()], -1); best_v, best_r, n_feas = -np.inf, None, 0
    for r0 in g:
        Rg = np.concatenate([np.full((rest.shape[0], 1), r0), rest], 1)
        o = M.evaluate(Rg, shapes, prm, True)
        v = np.asarray(o[objective], dtype=float)
        if cons is not None: v = np.where(cons(o), v, -np.inf)
        v = np.where(np.isfinite(o["nsv"]), v, -np.inf)
        n_feas += int(np.isfinite(v).sum())
        i = int(np.argmax(v))
        if v[i] > best_v: best_v, best_r = float(v[i]), Rg[i].copy()
    return best_r, best_v, n_feas

def decomposition(shapes, prm, extra=None, starts_k=14, n_starts=6, g_film_min=0):
    """Exact reformulation max_r NSV(r) = max_g max_{r in F_g} V_g(r); each subproblem solved
    by SLSQP from multiple grid starts. extra(o) returns additional slacks (>=0)."""
    gk = np.linspace(0, RMAX, starts_k); A, B, C, D = np.meshgrid(gk, gk, gk, gk, indexing="ij")
    Rs = np.stack([A.ravel(), B.ravel(), C.ravel(), D.ravel()], -1)
    os_ = M.evaluate(Rs, shapes, prm, True)
    best = dict(v=-np.inf); log = []
    for g in itertools.product((2, 1, 0), (2, 1, 0), (2, 1, 0)):
        if g[2] < g_film_min: continue
        sl = np.array(pattern_slacks(os_, g, prm)) if pattern_slacks(os_, g, prm) else np.zeros((0, Rs.shape[0]))
        if extra is not None: sl = np.vstack([sl, np.array(extra(os_))]) if sl.size else np.array(extra(os_))
        feas = np.all(sl >= 0, axis=0) if sl.size else np.ones(Rs.shape[0], bool)
        if not feas.any():
            log.append(dict(g=g, feasible_starts=0)); continue
        vv = np.where(feas, pattern_value(os_, g, prm), -np.inf)
        idx = np.argsort(-vv)[:n_starts]
        f = lambda r: -float(pattern_value(M.evaluate(r, shapes, prm, True), g, prm))
        def cfun(r):
            o = M.evaluate(r, shapes, prm, True); s = pattern_slacks(o, g, prm)
            if extra is not None: s = s + list(extra(o))
            return np.array([float(x) for x in s]) if s else np.array([1.0])
        cfun_m = lambda r: cfun(r) - DELTA          # small safety margin inside each grade region
        bg = dict(v=-np.inf)
        for i in idx:
            res = minimize(f, Rs[i], method="SLSQP", bounds=[(0, RMAX)] * 4,
                           constraints=[dict(type="ineq", fun=cfun_m)], options=dict(maxiter=300, ftol=1e-10))
            r = np.clip(res.x, 0, RMAX)
            v = -f(r)
            # accept only if the true (discontinuous) model confirms the pattern value
            if np.all(cfun(r) >= 0) and abs(float(M.evaluate(r, shapes, prm)) - v) < 1e-9:
                if v > bg["v"]: bg = dict(v=v, r=r)
            if vv[i] > bg["v"]: bg = dict(v=float(vv[i]), r=Rs[i])     # keep grid start if SLSQP fails
        log.append(dict(g=g, feasible_starts=int(feas.sum()), v=bg["v"]))
        if bg["v"] > best["v"]: best = dict(v=bg["v"], r=bg["r"], g=g)
    # true NSV at the solution must equal the pattern value (exactness check)
    best["nsv_true"] = float(M.evaluate(best["r"], shapes, prm))
    best["log"] = log
    return best

sh = M.shapes_for(FOCAL, "superellipse")
r_f1 = M.f1_policy(sh, "superellipse")
o_f1 = M.evaluate(r_f1, sh, BASEP, True); eff_f1 = float(o_f1["recovery_eff"])
rf_cons = lambda o: (o["recovery_eff"] >= eff_f1 - 1e-9) & (o["p_film"] > 0)

# ------------------------------------------------------------------ 1-2. enumeration timing and grid convergence
timing = {}
for k in ((14, 20) if TEST else (14, 20, 30, 50)):
    t = time.time(); r_u, v_u, _ = enum(sh, BASEP, k); tu = time.time() - t
    t = time.time(); r_c, v_c, nf = enum(sh, BASEP, k, cons=rf_cons); tc = time.time() - t
    timing[k] = dict(points=k ** 4, t_unc=tu, v_unc=v_u, r_unc=r_u.tolist(), t_rf=tc, v_rf=v_c, r_rf=r_c.tolist(), feasible_rf=nf)
    print("k", k, round(tu, 1), round(tc, 1), v_u, v_c, flush=True)
OUT["timing"] = timing

t = time.time(); dec_u = decomposition(sh, BASEP); t_du = time.time() - t
t = time.time()
dec_rf = decomposition(sh, BASEP, extra=lambda o: [o["recovery_eff"] - eff_f1], g_film_min=1)
t_drf = time.time() - t
OUT["decomposition"] = dict(
    unc=dict(v=dec_u["v"], nsv_true=dec_u["nsv_true"], r=np.asarray(dec_u["r"]).tolist(), g=dec_u["g"], t=t_du,
             patterns=[dict(g=l["g"], feasible_starts=l["feasible_starts"], v=l.get("v")) for l in dec_u["log"]]),
    rf=dict(v=dec_rf["v"], nsv_true=dec_rf["nsv_true"], r=np.asarray(dec_rf["r"]).tolist(), g=dec_rf["g"], t=t_drf,
            patterns=[dict(g=l["g"], feasible_starts=l["feasible_starts"], v=l.get("v")) for l in dec_rf["log"]]))
print("decomposition", dec_u["v"], dec_u["nsv_true"], dec_rf["v"], dec_rf["nsv_true"], time.time() - t0, flush=True)

# ------------------------------------------------------------------ 3. benchmark policies
def cost_sensitive(shapes, prm, grid=np.linspace(0.0, 0.99, 991)):
    """Grade-blind cost-sensitive thresholds: per class, maximise b_c*TP - kappa_c*FP with
    benefits and costs from nominal (clean-bale) prices; specifications are ignored."""
    pr = P.PRICES[prm["price"]]; m = prm["m"]; d = prm["disposal"]
    n = P.COUNT_SHARE * prm["w"] * 1000.0 / np.dot(P.COUNT_SHARE, m)
    p_metal = prm["al_share"] * pr["ubc"] + (1 - prm["al_share"]) * pr["steel"]
    nominal = np.array([pr["occ"], pr["film_a"], pr["rigid"], p_metal])
    b = m * (nominal - pr["mp"]) / 1000.0                       # USD gained per correct pick
    out = []
    for c in range(4):
        others = [j for j in range(4) if j != c]
        m_other = np.dot(n[others], m[others]) / n[others].sum()
        kappa = (prm["phi"] * prm["m_paper"] * (pr["mp"] + d) + (1 - prm["phi"]) * m_other * d) / 1000.0
        p = M.precision(grid, shapes[c], prm["pr_family"]); TP = grid * n[c]; FP = TP * (1 / p - 1)
        out.append(grid[int(np.argmax(b[c] * TP - kappa * FP))])
    return np.array(out), b

def purity_constrained(shapes, prm, k, pi=0.90):
    """Separation-design benchmark: maximise recovered target mass subject to every bin purity >= pi."""
    cons = lambda o: np.all((o["purity"] >= pi) | (o["bin_total"] <= 1e-9), axis=-1)
    r, v, _ = enum(shapes, prm, k, objective="recovered_total", cons=cons)
    return r

KEYS = ["nsv", "grade_line", "recovered_total", "misdirected", "paper_lost", "resid_nonfibre", "resid_cb",
        "contam_avoided", "recovery_eff", "spec_mass", "picks", "film_purity", "line_total", "purity", "recovered", "prohib", "outthrow"]
pack = lambda o: {kk: np.asarray(o[kk]).tolist() for kk in KEYS}
r_cs, b_cs = cost_sensitive(sh, BASEP)
r_pc = purity_constrained(sh, BASEP, 20 if TEST else 50)
OUT["benchmarks"] = {"Cost-sensitive (grade-blind)": dict(r=r_cs.tolist(), **pack(M.evaluate(r_cs, sh, BASEP, True))),
                     "Purity-constrained recovery max": dict(r=r_pc.tolist(), **pack(M.evaluate(r_pc, sh, BASEP, True)))}
OUT["cs_benefit_per_pick"] = b_cs.tolist()
print("benchmarks", {k_: v["nsv"] for k_, v in OUT["benchmarks"].items()}, time.time() - t0, flush=True)

# ------------------------------------------------------------------ 4. proposition checks (cliff location and jump)
def cliff(shapes, prm, r_base):
    """r_hat: smallest cardboard recall at which the paper product meets SRPN, others held at r_base."""
    def o_at(x):
        R = r_base.copy(); R[0] = x; return M.evaluate(R, shapes, prm, True)
    fine = np.linspace(0, RMAX, 4901); Rg = np.tile(r_base, (fine.size, 1)); Rg[:, 0] = fine
    og = M.evaluate(Rg, shapes, prm, True)
    out_share = np.asarray(og["outthrow"]); mono = bool(np.all(np.diff(out_share) <= 1e-12))
    ok = np.asarray(og["grade_line"]) == 2
    if not ok.any(): return dict(r_hat=None, monotone=mono)
    G = prm.get("grades", P.GRADES)
    fx = lambda x: float(o_at(x)["outthrow"]) - G["srpn_outthrow"]
    j = int(np.argmax(ok))
    r_hat = brentq(fx, fine[max(j - 1, 0)], fine[j]) if j > 0 and fx(fine[max(j - 1, 0)]) > 0 else float(fine[j])
    eps = 1e-6; lo, hi = o_at(max(r_hat - eps, 0)), o_at(min(r_hat + eps, RMAX))
    pr = P.PRICES[prm["price"]]
    pred = (pr["srpn"] - pr["mp"]) * float(hi["line_total"]) / 1000.0
    return dict(r_hat=float(r_hat), monotone=mono, jump=float(hi["nsv"] - lo["nsv"]), jump_pred=pred,
                line_total=float(hi["line_total"]), prohib_at=float(hi["prohib"]))
OUT["cliff_f1ctx"] = cliff(sh, BASEP, r_f1.copy())
print("cliff", OUT["cliff_f1ctx"], flush=True)

# ------------------------------------------------------------------ 5. paired Monte Carlo
rng = np.random.default_rng(20261015)
def draw():
    prm = dict(BASEP)
    prm["w"] = rng.uniform(*P.RANGES["w"]); prm["m"] = BASEP["m"] * np.exp(rng.normal(0, 0.35, 4))
    prm["m_paper"] = rng.uniform(*P.RANGES["m_paper"]); prm["phi"] = rng.uniform(*P.RANGES["phi"])
    prm["eta"] = rng.uniform(*P.RANGES["eta"]); prm["al_share"] = rng.uniform(*P.RANGES["al_share"])
    prm["disposal"] = rng.uniform(*P.RANGES["disposal"]); prm["price"] = "2025-06" if rng.random() < 0.5 else "2026-06"
    prm["pr_family"] = "power" if rng.random() < 0.5 else "superellipse"; return prm
NMC = 20 if TEST else 2000; KMC = 14; POL = ["F1", "CS", "PC", "GL", "RF"]
mc = {f"{a}_{p}": [] for a in ("gain", "srpn", "rec", "picks") for p in POL}
mc.update(dict(r_hat=[], rf1_cb=[], f1_below_cliff=[], prohib_f1=[], t_draw=[]))
for i in range(NMC):
    ti = time.time(); prm = draw(); shd = M.shapes_for(FOCAL, prm["pr_family"])
    none = float(M.evaluate(np.zeros(4), shd, prm))
    rf1 = M.f1_policy(shd, prm["pr_family"]); o1 = M.evaluate(rf1, shd, prm, True); ef = float(o1["recovery_eff"])
    rcs, _ = cost_sensitive(shd, prm)
    rpc = purity_constrained(shd, prm, KMC)
    rgl, _ = M.global_opt(shd, prm)
    rrf, vrf, _ = enum(shd, prm, KMC, cons=lambda o: (o["recovery_eff"] >= ef * 0.999) & (o["p_film"] > 0))
    if not np.isfinite(vrf): rrf = rf1
    for p_, r_ in zip(POL, (rf1, rcs, rpc, rgl, rrf)):
        o = M.evaluate(r_, shd, prm, True)
        mc["gain_" + p_].append(float(o["nsv"]) - none); mc["srpn_" + p_].append(int(o["grade_line"] == 2))
        mc["rec_" + p_].append(float(o["recovered_total"])); mc["picks_" + p_].append(float(o["picks"]))
    cl = cliff(shd, prm, rf1.copy())
    mc["r_hat"].append(cl["r_hat"]); mc["rf1_cb"].append(float(rf1[0]))
    mc["f1_below_cliff"].append(None if cl["r_hat"] is None else int(rf1[0] < cl["r_hat"]))
    mc["prohib_f1"].append(float(o1["prohib"])); mc["t_draw"].append(time.time() - ti)
    if i % 200 == 0: print("mc", i, round(time.time() - t0), flush=True)
OUT["mc"] = mc
json.dump(OUT, open("../results/results_caie_test.json" if TEST else "../results/results_caie.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else (list(o) if isinstance(o, tuple) else float(o)), indent=1)
print("saved", time.time() - t0)
