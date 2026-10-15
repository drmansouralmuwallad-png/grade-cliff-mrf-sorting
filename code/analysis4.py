"""Decomposition of the value gap between the tuned global threshold (GL) and the per-class
recovery-floor policy (VO-RF) in the paired Monte Carlo of analysis3.py.

The same 2,000 draws are regenerated (same seed and sampling order) and, for each draw:
  VO   : unconstrained per-class value optimum on the k=14 grid,
  RF   : per-class optimum with recovery floor and positive-value film (k=14), as in analysis3.py,
  GL   : one common recall tuned for value (99-point grid), as in analysis3.py,
plus whether the floor binds (VO already meets the floor) and whether the floor was infeasible
(RF falls back to F1). On every 10th draw RF is also solved on a k=30 grid to measure the
discretisation loss of k=14. Writes ../results/results_caie_gap.json.
"""
import json, time, numpy as np
import params as P, model as M

FOCAL = "GDINO-B (semi)"; BASEP = dict(P.BASE); RMAX = 0.98; t0 = time.time()

def enum(shapes, prm, k, cons=None):
    g = np.linspace(0, RMAX, k); A, B, C = np.meshgrid(g, g, g, indexing="ij")
    rest = np.stack([A.ravel(), B.ravel(), C.ravel()], -1); best_v, best_r = -np.inf, None
    for r0 in g:
        Rg = np.concatenate([np.full((rest.shape[0], 1), r0), rest], 1)
        o = M.evaluate(Rg, shapes, prm, True); v = np.asarray(o["nsv"], dtype=float)
        if cons is not None: v = np.where(cons(o), v, -np.inf)
        i = int(np.argmax(v))
        if v[i] > best_v: best_v, best_r = float(v[i]), Rg[i].copy()
    return best_r, best_v

rng = np.random.default_rng(20261015)          # identical to analysis3.py
def draw():
    prm = dict(BASEP)
    prm["w"] = rng.uniform(*P.RANGES["w"]); prm["m"] = BASEP["m"] * np.exp(rng.normal(0, 0.35, 4))
    prm["m_paper"] = rng.uniform(*P.RANGES["m_paper"]); prm["phi"] = rng.uniform(*P.RANGES["phi"])
    prm["eta"] = rng.uniform(*P.RANGES["eta"]); prm["al_share"] = rng.uniform(*P.RANGES["al_share"])
    prm["disposal"] = rng.uniform(*P.RANGES["disposal"]); prm["price"] = "2025-06" if rng.random() < 0.5 else "2026-06"
    prm["pr_family"] = "power" if rng.random() < 0.5 else "superellipse"; return prm

NMC = 2000; KMC = 14
out = {k: [] for k in ("w", "price", "family", "none", "gain_GL", "gain_RF", "gain_VO", "rec_GL", "rec_RF", "rec_VO",
                       "rec_F1", "floor_binds", "rf_fallback", "srpn_GL", "srpn_RF", "srpn_VO", "gain_RF30", "film_pos_VO")}
for i in range(NMC):
    prm = draw(); shd = M.shapes_for(FOCAL, prm["pr_family"])
    none = float(M.evaluate(np.zeros(4), shd, prm))
    rf1 = M.f1_policy(shd, prm["pr_family"]); o1 = M.evaluate(rf1, shd, prm, True); ef = float(o1["recovery_eff"])
    cons = lambda o: (o["recovery_eff"] >= ef * 0.999) & (o["p_film"] > 0)
    rgl, _ = M.global_opt(shd, prm)
    rvo, _ = enum(shd, prm, KMC)
    rrf, vrf = enum(shd, prm, KMC, cons=cons); fb = not np.isfinite(vrf)
    if fb: rrf = rf1
    ogl, ovo, orf = (M.evaluate(r, shd, prm, True) for r in (rgl, rvo, rrf))
    out["w"].append(prm["w"]); out["price"].append(prm["price"]); out["family"].append(prm["pr_family"]); out["none"].append(none)
    for k_, o in (("GL", ogl), ("RF", orf), ("VO", ovo)):
        out["gain_" + k_].append(float(o["nsv"]) - none); out["rec_" + k_].append(float(o["recovered_total"]))
        out["srpn_" + k_].append(int(o["grade_line"] == 2))
    out["rec_F1"].append(float(o1["recovered_total"]))
    out["floor_binds"].append(int(not bool(cons(ovo)))); out["rf_fallback"].append(int(fb))
    out["film_pos_VO"].append(int(float(ovo["p_film"]) > 0))
    if i % 10 == 0:
        r30, v30 = enum(shd, prm, 30, cons=cons)
        out["gain_RF30"].append(None if not np.isfinite(v30) else v30 - none)
    else:
        out["gain_RF30"].append(None)
    if i % 200 == 0: print("draw", i, round(time.time() - t0), flush=True)
json.dump(out, open("../results/results_caie_gap.json", "w"))
print("saved", round(time.time() - t0))
