"""Detection-to-bale-value model for a robotic quality-control station on an MRF paper line."""
import numpy as np
from scipy.special import gamma
from scipy.optimize import brentq
import params as P

# ---------------------------------------------------------------- PR-curve calibration
def superellipse_ap(a):
    return gamma(1 + 1 / a) ** 2 / gamma(1 + 2 / a)

def calibrate(ap, family):
    ap = float(np.clip(ap, 0.02, 0.98))
    if family == "superellipse":
        return brentq(lambda a: superellipse_ap(a) - ap, 0.05, 200.0)
    if family == "power":                # p = 1 - r^k, AP = k/(k+1)
        return ap / (1 - ap)
    raise ValueError(family)

def precision(r, shape, family):
    r = np.clip(r, 0.0, 0.999999)
    if family == "superellipse":
        return np.clip((1 - r ** shape) ** (1 / shape), 1e-6, 1.0)
    return np.clip(1 - r ** shape, 1e-6, 1.0)

def class_ap50(name, ap50_scale=None):
    ap, mAP, mAP50 = P.DETECTORS[name]
    if ap is None:
        ref = np.array(P.DETECTORS[P.PER_CLASS_REFERENCE][0])
        ap = ref / ref.mean() * mAP
    ap = np.array(ap, dtype=float)
    scale = (mAP50 / mAP) if ap50_scale is None else ap50_scale
    return np.clip(ap * scale / 100.0, 0.02, 0.98)

def shapes_for(name, family, ap50_scale=None):
    return np.array([calibrate(a, family) for a in class_ap50(name, ap50_scale)])

# ---------------------------------------------------------------- mass-flow + economics
def evaluate(R, shapes, prm, return_all=False):
    """R: array (..., 4) of per-class operating recalls. Returns NSV (USD/t feed) and flows."""
    R = np.asarray(R, dtype=float)
    fam = prm["pr_family"]; eta = prm["eta"]; phi = prm["phi"]
    m = prm["m"]; mp = prm["m_paper"]
    pr = P.PRICES[prm["price"]]; G = prm.get("grades", P.GRADES)
    MT = prm["w"] * 1000.0; MP = (1 - prm["w"]) * 1000.0
    n = P.COUNT_SHARE * MT / np.dot(P.COUNT_SHARE, m)          # items per tonne, per class

    prec = precision(R, shapes, fam)
    TP = R * n                                              # (...,4) true detections
    D = TP / prec                                           # all detections labelled c
    FP = D - TP
    # routing of false positives from other target classes
    W = np.tile(n, (4, 1)); np.fill_diagonal(W, 0.0); W = W / W.sum(axis=1, keepdims=True)  # W[c, c']
    FPo = (1 - phi) * FP[..., :, None] * W                  # (...,4 bin c,4 source c')
    FPp = phi * FP                                          # paper pieces per bin
    # item conservation: demand on each source class c'
    demand = TP + FPo.sum(axis=-2)                          # (...,4)
    s = np.minimum(1.0, n / np.maximum(demand, 1e-12))      # scale factor per source class
    TPs = TP * s
    FPos = FPo * s[..., None, :]
    picks = D.sum(axis=-1)                                  # attempted picks
    # bin contents (kg)
    bin_target = eta * TPs * m
    bin_other = eta * (FPos * m[None, :]).sum(axis=-1) if R.ndim > 1 else eta * (FPos * m[None, :]).sum(axis=-1)
    # separate prohibitive vs fibre content of each bin
    fibre_other = eta * FPos[..., :, 0] * m[0]              # cardboard mass entering bins 0..3
    nonfibre_other = bin_other - fibre_other
    bin_paper = eta * FPp * mp
    bin_total = bin_target + bin_other + bin_paper
    # paper line output
    removed = eta * (TPs + FPos.sum(axis=-2))               # items of each class removed
    resid = np.maximum(n - removed, 0.0) * m                # kg residual per class
    paper_out = MP - bin_paper.sum(axis=-1)
    line_total = paper_out + resid.sum(axis=-1)
    prohib = resid[..., 1:].sum(axis=-1) / line_total
    outthrow = resid[..., 0] / line_total
    p_line = np.where((prohib <= G["srpn_prohib"]) & (outthrow <= G["srpn_outthrow"]), pr["srpn"],
             np.where(prohib <= G["mp_prohib"], pr["mp"], prm["p_off"]))
    rev_line = line_total / 1000.0 * p_line
    grade_line = np.where((prohib <= G["srpn_prohib"]) & (outthrow <= G["srpn_outthrow"]), 2,
                 np.where(prohib <= G["mp_prohib"], 1, 0))
    # OCC bin (bin 0): target cardboard; prohibitives = non-fibre objects; outthrows = paper
    tot0 = np.maximum(bin_total[..., 0], 1e-12)
    pro0 = nonfibre_other[..., 0] / tot0
    out0 = bin_paper[..., 0] / tot0
    p_occ = np.where((pro0 <= G["occ_prohib"]) & (pro0 + out0 <= G["occ_total"]), pr["occ"],
            np.where(pro0 <= G["mp_prohib"], pr["mp"], prm["p_off"]))
    rev_occ = bin_total[..., 0] / 1000.0 * p_occ
    # film bin (bin 1): grade by purity
    tot1 = np.maximum(bin_total[..., 1], 1e-12)
    pur1 = bin_target[..., 1] / tot1
    p_film = np.where(pur1 >= G["film_a"], pr["film_a"], np.where(pur1 >= G["film_b"], pr["film_b"], pr["film_c"]))
    rev_film = bin_total[..., 1] / 1000.0 * p_film
    # rigid + metal bins: paid on target mass, disposal on the rest
    p_metal = prm["al_share"] * pr["ubc"] + (1 - prm["al_share"]) * pr["steel"]
    rev_rigid = bin_target[..., 2] / 1000.0 * pr["rigid"] - (bin_total[..., 2] - bin_target[..., 2]) / 1000.0 * prm["disposal"]
    rev_metal = bin_target[..., 3] / 1000.0 * p_metal - (bin_total[..., 3] - bin_target[..., 3]) / 1000.0 * prm["disposal"]
    nsv = rev_line + rev_occ + rev_film + rev_rigid + rev_metal - prm.get("c_pick", 0.0) * picks
    nsv = np.where(picks <= prm["budget"], nsv, -np.inf)
    if not return_all:
        return nsv
    # ---- physical (waste-management) indicators, kg per tonne of feed
    target_feed = n * m
    recovered = bin_target                                   # correct target mass in own bin
    misdirected = (bin_total - bin_target - bin_paper)       # target objects in a wrong bin
    paper_lost = bin_paper
    resid_nonfibre = resid[..., 1:].sum(axis=-1); resid_cb = resid[..., 0]
    none_nonfibre = target_feed[1:].sum(); none_cb = target_feed[0]
    # mass leaving at a specification grade with positive value
    spec_line = np.where(p_line > 0, line_total, 0.0)
    spec_occ = np.where(p_occ > 0, bin_total[..., 0], 0.0)
    spec_film = np.where(p_film > 0, bin_total[..., 1], 0.0)
    spec_mass = spec_line + spec_occ + spec_film + bin_target[..., 2] + bin_target[..., 3]
    downgraded = 1000.0 - spec_mass
    wm = dict(recovered=recovered, recovered_total=recovered.sum(axis=-1), misdirected=misdirected.sum(axis=-1),
              paper_lost=paper_lost.sum(axis=-1), resid_nonfibre=resid_nonfibre, resid_cb=resid_cb,
              contam_avoided=(none_nonfibre + none_cb) - (resid_nonfibre + resid_cb),
              recovery_eff=recovered.sum(axis=-1) / target_feed.sum(), spec_mass=spec_mass, downgraded=downgraded,
              line_total=line_total, paper_purity=paper_out / line_total, mass_check=line_total + bin_total.sum(axis=-1))
    pur = bin_target / np.maximum(bin_total, 1e-12)
    rec = bin_target / (n * m)
    return dict(nsv=nsv, rev_line=rev_line, rev_occ=rev_occ, rev_film=rev_film, rev_rigid=rev_rigid,
                rev_metal=rev_metal, prohib=prohib, outthrow=outthrow, grade_line=grade_line,
                occ_pro=pro0, occ_out=out0, film_purity=pur1, purity=pur, recovery=rec,
                picks=picks, n=n, bin_total=bin_total, bin_target=bin_target, resid=resid,
                p_occ=p_occ, p_film=p_film, prec=prec, p_line=p_line, bin_paper=bin_paper, **wm)

# ---------------------------------------------------------------- policies
def f1_policy(shapes, fam, grid=np.linspace(0.0, 0.99, 991)):
    r = []
    for sh in shapes:
        p = precision(grid, sh, fam)
        f1 = 2 * p * grid / np.maximum(p + grid, 1e-12)
        r.append(grid[np.argmax(f1)])
    return np.array(r)

def optimize(shapes, prm, k=50, rmax=0.98):
    """Exhaustive enumeration over a k^4 grid of per-class operating recalls (exact on the grid)."""
    g = np.linspace(0.0, rmax, k)
    A, B, C = np.meshgrid(g, g, g, indexing="ij")
    rest = np.stack([A.ravel(), B.ravel(), C.ravel()], axis=-1)
    best_v, best_r = -np.inf, None
    for r0 in g:
        Rg = np.concatenate([np.full((rest.shape[0], 1), r0), rest], axis=1)
        v = evaluate(Rg, shapes, prm)
        i = int(np.argmax(v))
        if v[i] > best_v:
            best_v, best_r = float(v[i]), Rg[i].copy()
    return best_r, best_v

def budget_scaled_f1(shapes, prm):
    r = f1_policy(shapes, prm["pr_family"])
    if np.isinf(prm["budget"]):
        return r
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ok = np.isfinite(evaluate(r * mid, shapes, prm))
        lo, hi = (mid, hi) if ok else (lo, mid)
    return r * lo


# ---------------------------------------------------------------- industrial baseline policies
def hp_policy(shapes, fam, pmin=0.90, grid=np.linspace(0.0, 0.99, 991)):
    """Fixed high-precision policy: largest recall with precision >= pmin, per class."""
    out = []
    for sh in shapes:
        p = precision(grid, sh, fam); ok = grid[p >= pmin]
        out.append(ok.max() if ok.size else 0.0)
    return np.array(out)

def uniform_policy(r, k=4):
    return np.full(k, r)

def global_opt(shapes, prm, grid=np.linspace(0, 0.98, 99)):
    """One global knob: common recall across classes, tuned for NSV."""
    v = evaluate(np.repeat(grid[:, None], 4, axis=1), shapes, prm)
    i = int(np.argmax(v)); return np.full(4, grid[i]), float(v[i])
