"""Checks that every operating point selected in the reported analyses is physically valid:
paper removed by false picks does not exceed the paper in the feed (equivalently, the paper product has
non-negative mass). The model does not cap paper removal, so extreme recalls with very low precision can
violate this under some parameter draws. Writes ../results/physical_check.json."""
import json, numpy as np, params as P, model as M
FOCAL = "GDINO-B (semi)"; BASEP = dict(P.BASE); RMAX = 0.98; OUT = {}
def phys(o, prm):
    return (np.asarray(o["line_total"]) >= -1e-9) & (np.asarray(o["bin_paper"]).sum(-1) <= (1 - prm["w"]) * 1000 + 1e-9)
def grid_share(sh, prm, k):
    g = np.linspace(0, RMAX, k); A, B, C, D = np.meshgrid(g, g, g, g, indexing="ij")
    R = np.stack([A.ravel(), B.ravel(), C.ravel(), D.ravel()], -1); o = M.evaluate(R, sh, prm, True)
    ph = phys(o, prm); v = np.asarray(o["nsv"])
    return float(1 - ph.mean()), bool((~ph).any() and v[~ph].max() >= v[ph].max() - 1e-9)

def make_draw(rng):
    def draw():
        prm = dict(BASEP)
        prm["w"] = rng.uniform(*P.RANGES["w"]); prm["m"] = BASEP["m"] * np.exp(rng.normal(0, 0.35, 4))
        prm["m_paper"] = rng.uniform(*P.RANGES["m_paper"]); prm["phi"] = rng.uniform(*P.RANGES["phi"])
        prm["eta"] = rng.uniform(*P.RANGES["eta"]); prm["al_share"] = rng.uniform(*P.RANGES["al_share"])
        prm["disposal"] = rng.uniform(*P.RANGES["disposal"]); prm["price"] = "2025-06" if rng.random() < 0.5 else "2026-06"
        prm["pr_family"] = "power" if rng.random() < 0.5 else "superellipse"; return prm
    return draw

# 1. first Monte Carlo (analysis2.py, seed 20260928): stored r_opt, r_rf; recompute F1
R2 = json.load(open("../results/results_revision.json")); mc = R2["mc"]; draw = make_draw(np.random.default_rng(20260928))
bad = dict(f1=0, opt=0, rf=0); unphys_grid = []; unphys_best = 0
for i in range(2000):
    prm = draw(); sh = M.shapes_for(FOCAL, prm["pr_family"])
    for key, r in (("f1", M.f1_policy(sh, prm["pr_family"])), ("opt", np.array(mc["r_opt"][i])), ("rf", np.array(mc["r_rf"][i]))):
        bad[key] += int(not phys(M.evaluate(r, sh, prm, True), prm))
    if i % 20 == 0:
        s, b = grid_share(sh, prm, 14); unphys_grid.append(s); unphys_best += int(b)
OUT["mc1"] = dict(unphysical_selected=bad, grid_sample_draws=len(unphys_grid), mean_unphysical_grid_share=float(np.mean(unphys_grid)),
                  max_unphysical_grid_share=float(np.max(unphys_grid)), draws_where_unphysical_point_beats_physical=unphys_best)
print("mc1", OUT["mc1"], flush=True)

# 2. paired Monte Carlo (analysis3/analysis4, seed 20261015): recompute each selected point
import importlib.util
spec = importlib.util.spec_from_file_location("a4", "analysis4.py")
src = open("analysis4.py").read().split("NMC = 2000")[0]           # helper definitions only
ns = {}; exec(src, ns)
enum = ns["enum"]; draw = make_draw(np.random.default_rng(20261015))
def cost_sensitive(shapes, prm, grid=np.linspace(0.0, 0.99, 991)):
    pr = P.PRICES[prm["price"]]; m = prm["m"]; d = prm["disposal"]
    n = P.COUNT_SHARE * prm["w"] * 1000.0 / np.dot(P.COUNT_SHARE, m)
    p_metal = prm["al_share"] * pr["ubc"] + (1 - prm["al_share"]) * pr["steel"]
    b = m * (np.array([pr["occ"], pr["film_a"], pr["rigid"], p_metal]) - pr["mp"]) / 1000.0; out = []
    for c in range(4):
        o_ = [j for j in range(4) if j != c]; mo = np.dot(n[o_], m[o_]) / n[o_].sum()
        kap = (prm["phi"] * prm["m_paper"] * (pr["mp"] + d) + (1 - prm["phi"]) * mo * d) / 1000.0
        p = M.precision(grid, shapes[c], prm["pr_family"]); TP = grid * n[c]; out.append(grid[int(np.argmax(b[c] * TP - kap * TP * (1 / p - 1)))])
    return np.array(out)
def purity_constrained(shapes, prm, k=14, pi=0.90):
    g = np.linspace(0, RMAX, k); A, B, C = np.meshgrid(g, g, g, indexing="ij"); rest = np.stack([A.ravel(), B.ravel(), C.ravel()], -1)
    bv, br = -np.inf, None
    for r0 in g:
        Rg = np.concatenate([np.full((rest.shape[0], 1), r0), rest], 1); o = M.evaluate(Rg, shapes, prm, True)
        v = np.where(np.all((o["purity"] >= pi) | (o["bin_total"] <= 1e-9), axis=-1) & np.isfinite(o["nsv"]), o["recovered_total"], -np.inf)
        i = int(np.argmax(v))
        if v[i] > bv: bv, br = float(v[i]), Rg[i].copy()
    return br
bad2 = dict(F1=0, CS=0, PC=0, GL=0, VO=0, RF=0); ug = []; ub = 0
for i in range(2000):
    prm = draw(); sh = M.shapes_for(FOCAL, prm["pr_family"])
    r1 = M.f1_policy(sh, prm["pr_family"]); ef = float(M.evaluate(r1, sh, prm, True)["recovery_eff"])
    cons = lambda o: (o["recovery_eff"] >= ef * 0.999) & (o["p_film"] > 0)
    rrf, vrf = enum(sh, prm, 14, cons=cons); rrf = r1 if not np.isfinite(vrf) else rrf
    for key, r in (("F1", r1), ("CS", cost_sensitive(sh, prm)), ("PC", purity_constrained(sh, prm)), ("GL", M.global_opt(sh, prm)[0]),
                   ("VO", enum(sh, prm, 14)[0]), ("RF", rrf)):
        bad2[key] += int(not phys(M.evaluate(r, sh, prm, True), prm))
    if i % 20 == 0:
        s, b = grid_share(sh, prm, 14); ug.append(s); ub += int(b)
    if i % 500 == 0: print("mc2", i, flush=True)
OUT["mc2"] = dict(unphysical_selected=bad2, grid_sample_draws=len(ug), mean_unphysical_grid_share=float(np.mean(ug)),
                  max_unphysical_grid_share=float(np.max(ug)), draws_where_unphysical_point_beats_physical=ub)
print("mc2", OUT["mc2"], flush=True)

# 3. deterministic analyses: baseline grids, scenarios (k=30), budgets, twelve detectors x two PR families (k=50 optima)
R = json.load(open("../results/results.json")); badd = 0; n = 0
for fam in ("superellipse", "power"):
    for d_, v in R[f"det_{fam}"].items():
        prm = dict(BASEP, pr_family=fam); sh = np.array(v["shapes"])
        for key in ("r_opt", "r_f1"):
            badd += int(not phys(M.evaluate(np.array(v[key]), sh, prm, True), prm)); n += 1
sh = M.shapes_for(FOCAL, "superellipse"); scen_unphys = []
STRICT = dict(P.GRADES, srpn_prohib=0.01, srpn_outthrow=0.02, mp_prohib=0.015)
for w in (0.04, 0.08, 0.14):
    for pr in ("2026-06", "2025-06"):
        for Gs in (P.GRADES, STRICT):
            prm = dict(BASEP, w=w, price=pr, grades=Gs); s, b = grid_share(sh, prm, 30); scen_unphys.append((w, pr, s, b))
OUT["deterministic"] = dict(detector_points_checked=n, unphysical=badd,
                            scenario_grid_unphysical_share_max=max(x[2] for x in scen_unphys), scenario_unphysical_beats=sum(x[3] for x in scen_unphys))
print("det", OUT["deterministic"])
json.dump(OUT, open("../results/physical_check.json", "w"), indent=1)
