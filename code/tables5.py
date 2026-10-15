"""Macros for the numerical checks of the propositions (math_check.json) and of physical validity
(physical_check.json)."""
import json, math
A = json.load(open("../results/math_check.json")); B = json.load(open("../results/physical_check.json"))
t2 = A["T2"]; t3 = A["T3_mc_f1_context"]; t3r = A["T3_random_context_baseline"]; rt = A["T3_r_tilde"]; t4 = A["T4"]
sel = sum(B["mc1"]["unphysical_selected"].values()) + sum(B["mc2"]["unphysical_selected"].values()) + B["deterministic"]["unphysical"]
nsel = 3 * 2000 + 6 * 2000 + B["deterministic"]["detector_points_checked"]
e = int(math.floor(math.log10(t2["max_abs_diff_physical"])))
mac = dict(
    CheckPts=f"{t2['physical_points']:,}", CheckEmpty=f"{t2['physical_points_with_empty_bin']:,}",
    CheckDev=r"$10^{%d}$" % (e + 1), CheckUnphys=str(t2["unphysical_points"]),
    PhysSel=f"{nsel:,}", PhysBad=str(sel),
    PhysShareMean=f"{100 * (B['mc1']['mean_unphysical_grid_share'] + B['mc2']['mean_unphysical_grid_share']) / 2:.1f}",
    PhysShareMax=f"{100 * max(B['mc1']['max_unphysical_grid_share'], B['mc2']['max_unphysical_grid_share']):.0f}",
    RtildeMin=f"{rt['min_mc']:.3f}", FOneCBMax=f"{rt['max_f1_cb_mc']:.3f}",
    CliffDraws=f"{t3['with_cliff']:,}", NoCliffDraws=str(t3["n"] - t3["with_cliff"]),
    SingleChange=f"{t3['single_paper_change']:,}", JumpErr=f"{t3['max_jump_err_after_smooth']:.3f}",
    CorAgree=f"{t4['agree']:,}", CorN=f"{t4['n']:,}", RhatMin=f"{t4['r_hat_min']:.3f}", RhatMax=f"{t4['r_hat_max']:.3f}",
    OtherChangeAtCliff=str(t3["other_change_exactly_at_cliff"]), RandomCtx=f"{t3r['n']:,}", RandomCliff=f"{t3r['with_cliff']:,}",
)
assert all(k.isalpha() for k in mac)
open("../tables/macros5.tex", "w").write("\n".join(r"\newcommand{\%s}{%s}" % kv for kv in mac.items()) + "\n")
print(mac)
