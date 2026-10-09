"""Input parameters with provenance. Every value is either (P) taken from a cited
public source or (A) an explicit modelling assumption that is varied in the
sensitivity and Monte Carlo analyses."""
import numpy as np

CLASSES = ["cardboard", "soft_plastic", "rigid_plastic", "metal"]
LABELS  = ["Cardboard", "Soft plastic", "Rigid plastic", "Metal"]

# (P) Item-count mix of the ZeroWaste stream: final pseudo-annotations on ZeroWaste-s,
# Abid et al. (2025), Table 5.
COUNTS = np.array([21352, 9806, 1523, 394], dtype=float)
COUNT_SHARE = COUNTS / COUNTS.sum()

# (P) Detector accuracy, Abid et al. (2025) Tables 1-4 (ZeroWaste-f test set, COCO metrics, %).
# per-class AP@[.5:.95] order: cardboard, soft plastic, rigid plastic, metal
DETECTORS = {
  # name: (per-class AP or None, mAP, mAP50, family, label)
  "GDINO-T (sup.)":  ([50.2, 47.8, 48.9, 36.6], 45.6, 56.8),
  "GDINO-T (semi)":  ([52.9, 52.9, 53.3, 38.1], 49.3, 60.3),
  "GDINO-B (sup.)":  ([54.6, 53.0, 57.0, 42.1], 51.6, 63.2),
  "GDINO-B (semi)":  ([55.8, 56.1, 61.2, 44.0], 54.3, 65.9),
  "Co-DETR Swin-L":  (None, 51.6, 63.0),
  "DETA Swin-L":     (None, 51.6, 62.6),
  "DINO Swin-L":     (None, 48.3, 59.8),
  "RT-DETR-L":       (None, 35.1, 45.1),
  "YOLO11-L":        (None, 34.0, 45.3),
  "RT-DETR-L (PL)":  (None, 39.4, 52.1),
  "YOLO11-L (PL)":   (None, 40.3, 51.1),
  "TridentNet":      (None, 24.2, 36.3),
}
PER_CLASS_REFERENCE = "GDINO-B (sup.)"   # (A) class pattern borrowed for mAP-only detectors

# ---------------- Prices (P): SMP Index national averages, Resource Recycling, 15 June 2026
LB_PER_T = 2204.62            # lb per metric tonne
ST_PER_T = 1.0 / 0.907185     # short tons per metric tonne
def cents_lb(c): return c / 100.0 * LB_PER_T
def usd_st(u):   return u * ST_PER_T

PRICES = {
 "2026-06": dict(
    srpn=usd_st(42.5), mp=usd_st(29.0), occ=usd_st(72.5),
    film_a=cents_lb(15.0), film_b=cents_lb(4.0), film_c=cents_lb(-0.56),
    rigid=cents_lb(13.13), ubc=cents_lb(93.0), steel=usd_st(200.0)),
 # same source, "one year ago" values; film B/C and OCC not reported for 2025 -> held at 2026 values
 "2025-06": dict(
    srpn=usd_st(56.0), mp=usd_st(39.0), occ=usd_st(72.5),
    film_a=cents_lb(19.06), film_b=cents_lb(4.0), film_c=cents_lb(-0.56),
    rigid=cents_lb(4.13), ubc=cents_lb(83.75), steel=usd_st(168.0)),
}

# ---------------- Grade rules (P): ReMA/ISRI paper-stock specifications; APR PE-film model bale specs
GRADES = dict(
    srpn_prohib=0.02, srpn_outthrow=0.03,   # Sorted Residential Papers: prohibitives<=2 %, outthrows<=3 %
    mp_prohib=0.02,                         # Mixed Paper: prohibitives<=2 %
    occ_prohib=0.01, occ_total=0.05,        # OCC #11: prohibitives<=1 %, outthrows+prohibitives<=5 %
    film_a=0.95, film_b=0.90)               # APR clear film Grade A <=5 %, Grade B <=10 % contamination

# ---------------- Baseline modelling assumptions (A) with ranges used in sensitivity analysis
BASE = dict(
    w=0.08,                     # target-object mass share of the paper-line feed
    m=np.array([50., 10., 30., 25.]) / 1000.0,   # mean item mass, kg (CB, SP, RP, M)
    m_paper=15.0 / 1000.0,      # mass of a paper piece picked by a false positive, kg
    phi=0.5,                    # share of false positives originating from paper background
    eta=1.0,                    # mechanical pick success given a detection
    al_share=0.5,               # aluminium share of the metal class (rest steel)
    disposal=50.0,              # USD/t charged on non-target mass in plastic/metal bales
    p_off=0.0,                  # USD/t for an off-specification paper bale
    ap50_scale=None,            # None -> use detector mAP50/mAP ratio
    pr_family="superellipse",
    price="2026-06",
    budget=np.inf,              # attempted picks per tonne of feed
)
RANGES = dict(
    w=(0.03, 0.15),
    m_scale=(0.5, 2.0),
    m_paper=(0.005, 0.030),
    phi=(0.2, 0.8),
    eta=(0.85, 1.0),
    al_share=(0.2, 0.8),
    disposal=(25.0, 100.0),
    p_off=(-20.0, 20.0),
)
