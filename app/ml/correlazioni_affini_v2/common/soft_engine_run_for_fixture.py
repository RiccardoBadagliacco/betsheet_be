import numpy as np
import pandas as pd

def run_soft_engine_for_fixture(
    target_row: pd.Series,
    clusters: dict,
    slim_index: pd.DataFrame,
    wide_index: pd.DataFrame,
    top_n=80,
    min_neighbors=20,
):
    """
    Soft engine per fixture.
    Il target non è presente nello SLIM → filtriamo per cluster e feature.
    """

    c1 = clusters["cluster_1x2"]

    cand = slim_index[slim_index["cluster_1x2"] == c1].copy()

    if len(cand) < min_neighbors:
        return {"status": "error", "reason": "not_enough_candidates"}

    key_cols = [
        "elo_diff", "lambda_total_form", "market_sharpness",
        "delta_p1", "delta_p2", "delta_O25", "delta_U25"
    ]
    key_cols = [c for c in key_cols if c in cand.columns]

    X = cand[key_cols].values
    t = target_row[key_cols].values
    d = np.sqrt(((X - t) ** 2).sum(axis=1))
    cand["distance"] = d

    cand = cand.sort_values("distance").head(top_n)
    w = np.exp(-2 * (cand["distance"] / np.percentile(d, 90)))
    cand["weight"] = w
    wsum = w.sum()

    merge = wide_index.merge(cand[["match_id", "weight"]], on="match_id")
    gh = merge["home_ft"].values
    ga = merge["away_ft"].values
    wt = merge["weight"].values / wsum

    return {
        "status": "ok",
        "soft_probs": {
            "p1": float((wt * (gh > ga)).sum()),
            "px": float((wt * (gh == ga)).sum()),
            "p2": float((wt * (gh < ga)).sum()),
            "pO15": float((wt * ((gh + ga) >= 2)).sum()),
            "pU15": float((wt * ((gh + ga) < 2)).sum()),
            "pO25": float((wt * ((gh + ga) >= 3)).sum()),
            "pU25": float((wt * ((gh + ga) < 3)).sum()),
        },
        "affini_stats": {
            "n_affini_soft": len(cand),
            "avg_distance": float(cand["distance"].mean()),
        },
        "config_used": {"top_n": top_n, "min_neighbors": min_neighbors},
    }