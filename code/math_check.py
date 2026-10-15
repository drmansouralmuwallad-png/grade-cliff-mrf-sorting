"""Numerical checks of the assumptions and statements of Lemma 1, Proposition 1, Proposition 2 and Corollary 1.

T1  Nested grade limits and nondecreasing prices (all price years, off-spec price range).
T2  Proposition 1: Pi(r) == max{V_g(r) : g in Gamma, r in F_g} at random operating points over random draws.
T3  Proposition 2 along the cardboard recall, in the F1 context of each of the 2,000 paired Monte Carlo draws
    and in 2,000 random contexts at baseline: monotone outthrow share, other SRPN/MP limits hold over the
    whole sweep, a single paper-grade change, jump height equal to the closed form; other streams' grade
    changes along the sweep are counted (they are the reason Prop. 2(ii) is stated per stream).
T4  Corollary 1: F1 keeps SRPN iff its cardboard recall >= r_hat (re-derived independently of analysis3.py).
Writes ../results/math_check.json.
"""
import json, itertools, time, numpy as np
import params as P, model as M
t0 = time.time(); OUT = {}; FOCAL = "GDINO-B (semi)"; BASEP = dict(P.BASE)

# ---------------- T1
t1 = {}
G = P.GRADES
t1["paper_nested"] = bool(G["srpn_prohib"] <= G["mp_prohib"])          # SRPN limits imply MP limit
t1["occ_nested"] = bool(G["occ_prohib"] <= G["mp_prohib"])             # OCC limits imply MP-grade limit for the bin
t1["film_nested"] = bool(G["film_a"] >= G["film_b"])
mono = []
for py, pr in P.PRICES.items():
    for p_off in (P.RANGES["p_off"][0], 0.0, P.RANGES["p_off"][1]):
        mono.append(dict(price=py, p_off=p_off, paper=p_off <= pr["mp"] <= pr["srpn"], occ=p_off <= pr["mp"] <= pr["occ"],
                         film=pr["film_c"] <= pr["film_b"] <= pr["film_a"]))
t1["price_monotone_all"] = all(m["paper"] and m["occ"] and m["film"] for m in mono); t1["cases"] = mono
OUT["T1"] = t1; print("T1", t1["paper_nested"], t1["occ_nested"], t1["film_nested"], t1["price_monotone_all"], flush=True)

# ---------------- helpers
def prices(prm):
    pr = P.PRICES[prm["price"]]
    return ({2: pr["srpn"], 1: pr["mp"], 0: prm["p_off"]}, {2: pr["occ"], 1: pr["mp"], 0: prm["p_off"]}, {2: pr["film_a"], 1: pr["film_b"], 0: pr["film_c"]})
def feasible(o, g, Gr):
    ok = np.ones(np.shape(o["nsv"]), bool)
    if g[0] >= 1: ok &= o["prohib"] <= Gr["mp_prohib"]
    if g[0] == 2: ok &= (o["prohib"] <= Gr["srpn_prohib"]) & (o["outthrow"] <= Gr["srpn_outthrow"])
    if g[1] >= 1: ok &= o["occ_pro"] <= Gr["mp_prohib"]
    if g[1] == 2: ok &= (o["occ_pro"] <= Gr["occ_prohib"]) & (o["occ_pro"] + o["occ_out"] <= Gr["occ_total"])
    if g[2] >= 1: ok &= o["film_purity"] >= Gr["film_b"]
    if g[2] == 2: ok &= o["film_purity"] >= Gr["film_a"]
    return ok
def V(o, g, prm):
    PL, PO, PF = prices(prm)
    return (o["line_total"] * PL[g[0]] + o["bin_total"][..., 0] * PO[g[1]] + o["bin_total"][..., 1] * PF[g[2]]) / 1000.0 \
        + o["rev_rigid"] + o["rev_metal"] - prm.get("c_pick", 0.0) * o["picks"]

rng = np.random.default_rng(20261015)
def draw():
    prm = dict(BASEP)
    prm["w"] = rng.uniform(*P.RANGES["w"]); prm["m"] = BASEP["m"] * np.exp(rng.normal(0, 0.35, 4))
    prm["m_paper"] = rng.uniform(*P.RANGES["m_paper"]); prm["phi"] = rng.uniform(*P.RANGES["phi"])
    prm["eta"] = rng.uniform(*P.RANGES["eta"]); prm["al_share"] = rng.uniform(*P.RANGES["al_share"])
    prm["disposal"] = rng.uniform(*P.RANGES["disposal"]); prm["price"] = "2025-06" if rng.random() < 0.5 else "2026-06"
    prm["pr_family"] = "power" if rng.random() < 0.5 else "superellipse"; return prm
draws = [draw() for _ in range(2000)]          # identical to the paired Monte Carlo draws

def phys(o, prm):
    return (np.asarray(o["line_total"]) >= -1e-9) & (np.asarray(o["bin_paper"]).sum(-1) <= (1 - prm["w"]) * 1000 + 1e-9)
# ---------------- T2: Proposition 1 identity
rs = np.random.default_rng(7); maxdiff = 0.0; npts = 0; nzero = 0
STRICT = dict(P.GRADES, srpn_prohib=0.01, srpn_outthrow=0.02, mp_prohib=0.015)
for i in range(200):
    prm = dict(draws[i]); prm["p_off"] = rs.uniform(*P.RANGES["p_off"])
    if i % 2: prm["grades"] = STRICT
    sh = M.shapes_for(FOCAL, prm["pr_family"])
    R = rs.uniform(0, 0.98, (2000, 4)); R[rs.random((2000, 4)) < 0.1] = 0.0     # include empty bins
    o = M.evaluate(R, sh, prm, True); Gr = prm.get("grades", P.GRADES)
    best = np.full(R.shape[0], -np.inf)
    for g in itertools.product((2, 1, 0), repeat=3):
        best = np.maximum(best, np.where(feasible(o, g, Gr), V(o, g, prm), -np.inf))
    ph = phys(o, prm); d = np.abs(best - o["nsv"])
    maxdiff = max(maxdiff, float(d[ph].max())); npts += int(ph.sum()); nzero += int(((R == 0).any(1) & ph).sum())
    maxdiff_unphys = max(globals().get("maxdiff_unphys", 0.0), float(d[~ph].max()) if (~ph).any() else 0.0); globals()["maxdiff_unphys"] = maxdiff_unphys
    globals()["n_unphys"] = globals().get("n_unphys", 0) + int((~ph).sum())
OUT["T2"] = dict(physical_points=npts, physical_points_with_empty_bin=nzero, max_abs_diff_physical=maxdiff, unphysical_points=n_unphys, max_abs_diff_unphysical=maxdiff_unphys)
print("T2", OUT["T2"], round(time.time() - t0), flush=True)

# ---------------- T3/T4: Proposition 2 and Corollary 1 along the cardboard recall
grid = np.linspace(0, 0.98, 4901)
def sweep(sh, prm, rbase):
    Rg = np.tile(rbase, (grid.size, 1)); Rg[:, 0] = grid
    return M.evaluate(Rg, sh, prm, True)
def check(o, prm, r1cb=None):
    Gr = prm.get("grades", P.GRADES); pr = P.PRICES[prm["price"]]
    ph = phys(o, prm); kmax = int(np.argmin(ph)) if (~ph).any() else len(ph)       # physical part of the sweep
    s_all = np.asarray(o["outthrow"])[:kmax]; inc = np.where(np.diff(s_all) > 1e-12)[0]
    kmax = int(inc[0]) + 1 if inc.size else kmax                                    # interval [0, r_tilde] with nonincreasing share
    o = {k_: (np.asarray(v)[:kmax] if np.ndim(v) >= 1 and np.shape(v)[0] == len(ph) else v) for k_, v in o.items()}
    s = np.asarray(o["outthrow"]); pro = np.asarray(o["prohib"]); gl = np.asarray(o["grade_line"])
    res = dict(r_tilde=float(grid[kmax - 1]), physical_upto=float(grid[kmax - 1]), truncated=bool(kmax < len(ph)), monotone=bool(np.all(np.diff(s) <= 1e-12)), other_limits_hold=bool(np.all(pro <= Gr["srpn_prohib"])),
               starts_above=bool(s[0] > Gr["srpn_outthrow"]), ends_below=bool(s[-1] <= Gr["srpn_outthrow"]),
               paper_grade_changes=int(np.sum(np.diff(gl) != 0)),
               other_stream_changes=int(np.sum(np.diff(np.asarray(o["p_occ"])) != 0) + np.sum(np.diff(np.asarray(o["p_film"])) != 0)))
    if res["starts_above"] and res["ends_below"]:
        j = int(np.argmax(gl == 2)); res["r_hat"] = float(grid[j])
        res["jump"] = float(o["nsv"][j] - o["nsv"][j - 1])
        # closed form at the cliff, plus the smooth change over one grid step (computed with the grade held fixed)
        res["jump_pred"] = (pr["srpn"] - pr["mp"]) * float(o["line_total"][j]) / 1000.0
        sm_hi = float(o["nsv"][j]) - pr["srpn"] * float(o["line_total"][j]) / 1000.0
        sm_lo = float(o["nsv"][j - 1]) - pr["mp"] * float(o["line_total"][j - 1]) / 1000.0
        res["smooth_step"] = sm_hi - sm_lo
        res["other_change_at_cliff"] = bool(o["p_occ"][j] != o["p_occ"][j - 1] or o["p_film"][j] != o["p_film"][j - 1])
        if r1cb is not None: res["f1_keeps"] = None
    return res
t3 = []
for i, prm in enumerate(draws):
    sh = M.shapes_for(FOCAL, prm["pr_family"]); r1 = M.f1_policy(sh, prm["pr_family"])
    c = check(sweep(sh, prm, r1), prm)
    o1 = M.evaluate(r1, sh, prm, True)
    c["f1_keeps"] = int(o1["grade_line"] == 2); c["f1_cb"] = float(r1[0])
    t3.append(c)
    if i % 500 == 0: print("T3", i, round(time.time() - t0), flush=True)
# random contexts at baseline
t3r = []
sh0 = M.shapes_for(FOCAL, "superellipse")
for i in range(2000):
    rbase = rs.uniform(0, 0.98, 4); t3r.append(check(sweep(sh0, BASEP, rbase), BASEP))
def summ(L):
    cl = [c for c in L if "r_hat" in c]
    return dict(n=len(L), monotone=sum(c["monotone"] for c in L), other_limits_hold=sum(c["other_limits_hold"] for c in L),
                with_cliff=len(cl), single_paper_change=sum(c["paper_grade_changes"] == 1 for c in cl),
                max_jump_err_after_smooth=max(abs(c["jump"] - c["jump_pred"] - c["smooth_step"]) for c in cl) if cl else None,
                max_smooth_step=max(abs(c["smooth_step"]) for c in cl) if cl else None,
                sweeps_with_other_stream_changes=sum(c["other_stream_changes"] > 0 for c in L),
                other_change_exactly_at_cliff=sum(c["other_change_at_cliff"] for c in cl),
                truncated_sweeps=sum(c["truncated"] for c in L), starts_above=sum(c["starts_above"] for c in L), ends_below=sum(c["ends_below"] for c in L))
OUT["T3_mc_f1_context"] = summ(t3); OUT["T3_random_context_baseline"] = summ(t3r)
# Corollary: among draws satisfying the proposition's conditions
ok = [c for c in t3 if "r_hat" in c and c["monotone"] and c["other_limits_hold"]]
OUT["T4_failing_conditions"] = dict(no_cliff=sum("r_hat" not in c for c in t3), not_monotone=sum(not c["monotone"] for c in t3), other_limits_fail=sum(not c["other_limits_hold"] for c in t3))
print(OUT["T4_failing_conditions"], flush=True)
if not ok: json.dump(OUT, open("../results/math_check.json", "w"), indent=1, default=float); raise SystemExit("no draws satisfy conditions")
OUT["T3_r_tilde"] = dict(min_mc=min(c["r_tilde"] for c in t3), min_random=min(c["r_tilde"] for c in t3r),
                        max_r_hat_mc=max(c["r_hat"] for c in t3 if "r_hat" in c), max_f1_cb_mc=max(c["f1_cb"] for c in t3),
                        draws_f1_beyond_r_tilde=sum(c["f1_cb"] > c["r_tilde"] for c in t3))
print("r_tilde", OUT["T3_r_tilde"], flush=True)
OUT["T4"] = dict(n=len(ok), agree=sum(int(c["f1_cb"] >= c["r_hat"] - 1e-9) == c["f1_keeps"] for c in ok),
                 r_hat_min=min(c["r_hat"] for c in ok), r_hat_max=max(c["r_hat"] for c in ok),
                 f1_cb_min=min(c["f1_cb"] for c in ok), f1_cb_max=max(c["f1_cb"] for c in ok))
print(json.dumps({k: v for k, v in OUT.items() if k != "T1"}, indent=1)); print("time", round(time.time() - t0))
json.dump(OUT, open("../results/math_check.json", "w"), indent=1, default=float)
