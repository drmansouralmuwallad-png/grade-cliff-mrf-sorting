# Code and results for "Crossing the grade cliff: Specification-aware threshold optimisation for AI-guided robotic sorting at material recovery facilities"

Author: Mansour Almuwallad (ORCID 0009-0002-9547-0621), Department of Management Science, Yanbu Industrial College,
Royal Commission for Jubail and Yanbu, Saudi Arabia.

This archive reproduces every number, table and figure in the article and its Supplementary Material.

## Contents
- code/params.py: all model parameters, prices, grade rules, detector inputs, ranges for sensitivity and Monte Carlo analysis
- code/model.py: detector error model (calibrated precision–recall curves), robotic separation, mass balance, grading and net sales value
- code/analysis.py: detector comparison, pick budgets, class weights, sweeps, film frontier, one-at-a-time sensitivity
- code/analysis2.py: seven operating policies, Pareto cloud, twelve transferability scenarios, cost per pick, 2,000-draw Monte Carlo, class-weight intervals
- code/tables.py, code/tables2.py: write every table and number macro used in the manuscript (no hand-typed numbers)
- code/figures.py, figures2.py, fig1_framework.py, fig5_map_vs_value.py, fig9_sankey.py: intermediate figures
- code/figures_jem.py: final Figs 1–5 and Figs S1–S4
- results/: results.json, results_revision.json, fig3_numbers.json (outputs of the analysis scripts)
- tables/: generated LaTeX tables and macros; tables/supp/ holds Tables S1–S8; Table 1 (comparison with literature) is tables/tab_gap.tex
- figures_jem/: final figures (PDF, SVG, TIFF); figures_intermediate/: figures from the earlier scripts
- CAIE version (Computers & Industrial Engineering):
  - code/analysis3.py: computational study, benchmark policies, cliff check, paired Monte Carlo
  - code/analysis4.py: decomposition of the gap between the global threshold (GL) and VO-RF on the same draws
  - code/math_check.py: numerical checks of Lemma 1, Propositions 1-2 and Corollary 1
  - code/physical_check.py: physical validity (non-negative masses) of every selected operating point
  - code/tables3.py, tables4.py, tables5.py: CAIE tables and number macros
  - code/figures_caie.py, fig9_sankey_caie.py, graphical_abstract.py: CAIE figures and graphical abstract
  - results/results_caie.json, results_caie_gap.json, math_check.json, physical_check.json, analysis3.log, analysis4.log
  - figures_caie/: final CAIE figures (PDF, SVG, TIFF)

## Scenario definitions
Baseline and ranges: code/params.py (BASE, RANGES, PRICES, grade rules). Twelve scenarios: three target mass shares
(4%, 8%, 14%) x two price years (June 2025, June 2026) x two specification environments (standard; strict:
SRPN prohibitives <= 1% and outthrows <= 2%, mixed paper prohibitives <= 1.5%), defined in code/analysis2.py.

## Reproduce
Python 3 with numpy, scipy, matplotlib, adjustText.
    cd code
    python3 analysis.py && python3 analysis2.py       # about 10 min on one CPU; rewrites results/*.json
    python3 figures.py && python3 figures2.py         # intermediate figures
    python3 tables.py && python3 tables2.py           # tables and macros
    python3 figures_jem.py                            # final figures
The stored results/*.json let the tables and figures be rebuilt without re-running the analysis.

CAIE version (Computers & Industrial Engineering):
    cd code
    python3 analysis3.py        # computational study, benchmarks, cliff check, paired Monte Carlo (~5 min)
    python3 analysis4.py        # GL vs VO-RF gap decomposition on the same draws (~8 min)
    python3 math_check.py       # numerical checks of Lemma 1, Propositions 1-2, Corollary 1 (~1 min)
    python3 physical_check.py   # physical validity of every selected operating point (~10 min)
    python3 tables3.py && python3 tables4.py && python3 tables5.py
    python3 figures_caie.py && python3 fig9_sankey_caie.py   # fig9_sankey_caie.py needs: pip install adjustText
Random draws use the fixed seed 20261015, so the outputs are identical on every run (apart from timings in the logs).

## Inputs and their sources
All inputs are public: ZeroWaste annotation counts and detector accuracies as reported by Abid et al. (2025,
arXiv:2508.18799); ZeroWaste dataset doi:10.5281/zenodo.4899926 (CC BY-NC 4.0; no images or annotations are
redistributed here, only published summary statistics); bale specifications from ReMA (paper) and APR (PE film);
national average prices from the Secondary Materials Pricing index (Resource Recycling, 15 June 2026).

## Licence
Code: MIT licence (LICENSE). Results, tables and figures: CC BY 4.0.
