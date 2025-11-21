import numpy as np
import math
from typing import Dict, Any, List


# ================================================================
# Helper
# ================================================================
def normalize(x):
    s = float(sum(x))
    if s <= 0:
        return [0.0] * len(x)
    return [float(v) / s for v in x]


# ================================================================
# Estensione Poisson della PMF (smoothing)
# ================================================================
def smooth_pmf(pmf: Dict[int, float], max_goals=5) -> Dict[int, float]:
    """
    Se gli affini non hanno casi >=4 gol, la coda Poisson copre.
    λ = media pesata dei gol (fino a max_goals).
    """
    # media goals dal pmf troncato
    lam = sum(k * pmf[k] for k in pmf)

    tail = {}
    for k in range(0, max_goals + 1):
        pk = math.exp(-lam) * lam**k / math.factorial(k)
        tail[k] = pk

    # mescola affini reali con poisson
    final = {}
    for k in range(0, max_goals + 1):
        if pmf[k] > 0:
            final[k] = pmf[k]
        else:
            final[k] = 0.5 * tail[k]   # smoothing leggero

    final = normalize(list(final.values()))
    return {i: final[i] for i in range(0, max_goals + 1)}


# ================================================================
# PMF dagli affini REALI + smoothing
# ================================================================
def compute_pmf_from_affini(affini, max_goals=5):
    if not affini:
        uniform = {k: 1/(max_goals+1) for k in range(max_goals+1)}
        return {"home": uniform, "away": uniform}

    w = np.array([float(a.get("weight", 1.0)) for a in affini])

    def safe_int(val):
        try:
            if val is None:
                return 0
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return 0
            return int(val)
        except:
            return 0

    gh = np.array([safe_int(a.get("home_ft", 0)) for a in affini])
    ga = np.array([safe_int(a.get("away_ft", 0)) for a in affini])

    def _pmf(goals):
        pmf = {}
        w_sum = float(w.sum())
        for k in range(0, max_goals+1):
            pmf[k] = float(((goals == k).astype(float) * w).sum() / w_sum)
        return pmf

    pmf_home = smooth_pmf(_pmf(gh), max_goals=max_goals)
    pmf_away = smooth_pmf(_pmf(ga), max_goals=max_goals)

    return {"home": pmf_home, "away": pmf_away}


# ================================================================
# Top scores
# ================================================================
def compute_top_scores(pmf_home, pmf_away, max_goals=5):
    scores = []
    for h in range(max_goals+1):
        for a in range(max_goals+1):
            scores.append({
                "home_goals": h,
                "away_goals": a,
                "prob": pmf_home[h] * pmf_away[a]
            })
    return sorted(scores, key=lambda x: x["prob"], reverse=True)[:3]


# ================================================================
# Multigoal
# ================================================================
def multigoal_range(pmf, a, b):
    return float(sum(pmf[k] for k in range(a, b+1)))


# ================================================================
# Summary
# ================================================================
def build_summary(meta, clusters, prob, stats):
    return (
        f"🔍 Analisi {meta['home_team']} – {meta['away_team']} "
        f"({meta.get('league_name','')})\n"
        f"Cluster 1X2={clusters['cluster_1x2']}, "
        f"OU25={clusters['cluster_ou25']}, "
        f"OU15={clusters['cluster_ou15']}.\n"
        f"P1 {prob['p1']*100:.1f}%, PX {prob['px']*100:.1f}%, "
        f"P2 {prob['p2']*100:.1f}%.\n"
        f"Under2.5 stimato {prob['pU25']*100:.1f}%.\n"
        f"Affini utilizzati {stats['n_affini_soft']}, "
        f"distanza media {stats['avg_distance']:.2f}."
    )


# ================================================================
# POSTPROCESS COMPLETO (versione FIXATA)
# ================================================================
def full_postprocess(data: Dict[str, Any]) -> Dict[str, Any]:

    meta     = data["meta"]
    clusters = data["clusters"]
    prob     = data["soft_probs"]
    stats    = data["affini_stats"]
    affini   = data.get("affini") or data.get("affini_list") or []

    # -------- PMF DAGLI AFFINI VERE --------
    pmf_all = compute_pmf_from_affini(affini, max_goals=5)
    pmf_home = pmf_all["home"]
    pmf_away = pmf_all["away"]

    # -------- Segna Casa / Ospite COERENTE --------
    home_yes = 1 - pmf_home[0]
    away_yes = 1 - pmf_away[0]

    home_no = 1 - home_yes
    away_no = 1 - away_yes

    # -------- GG/NG COERENTE --------
    gg = home_yes * away_yes
    ng = 1 - gg

    # -------- Multigoal --------
    mg = {
        "home_1_3": multigoal_range(pmf_home, 1, 3),
        "home_1_4": multigoal_range(pmf_home, 1, 4),
        "home_1_5": multigoal_range(pmf_home, 1, 5),
        "away_1_3": multigoal_range(pmf_away, 1, 3),
        "away_1_4": multigoal_range(pmf_away, 1, 4),
        "away_1_5": multigoal_range(pmf_away, 1, 5),
    }

    # totale goals
    pmf_tot = normalize([
        sum(pmf_home[h] * pmf_away[a] for h in range(6) for a in range(6) if h+a==k)
        for k in range(11)
    ])

    mg["tot_1_3"] = sum(pmf_tot[1:4])
    mg["tot_1_4"] = sum(pmf_tot[1:5])
    mg["tot_1_5"] = sum(pmf_tot[1:6])

    # -------- Top scores --------
    top_scores = compute_top_scores(pmf_home, pmf_away)

    # -------- Summary --------
    summary = build_summary(meta, clusters, prob, stats)

    # -------- PACK FINALE --------
    return {
        "summary": summary,
        "markets": {
            "1x2": {
                "p1": prob["p1"],
                "px": prob["px"],
                "p2": prob["p2"],
            },
            "ou": {
                "o15": prob["pO15"],
                "u15": prob["pU15"],
                "o25": prob["pO25"],
                "u25": prob["pU25"],
            },
            "gg_ng": {
                "gg": gg,
                "ng": ng,
            },
            "score_team": {
                "home_yes": home_yes,
                "home_no": home_no,
                "away_yes": away_yes,
                "away_no": away_no,
            },
            "multigoal": mg,
            "pmf": {
                "home": pmf_home,
                "away": pmf_away,
            },
            "top_scores": top_scores,
        },
    }