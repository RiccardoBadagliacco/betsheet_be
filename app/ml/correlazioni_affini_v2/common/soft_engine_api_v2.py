# ============================================================
# soft_engine_api_v2.py  — VERSIONE API STABILE
# ============================================================
# Funzione principale: run_soft_engine_api(target_row, slim, wide)
# Restituisce clusters, soft_probs, affini_stats, affini_list
# Compatibile con API /matches/{id} e /fixtures/{id}
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# 1. DISTANZA NORMALIZZATA (EUCLIDEA Z-SCORE)
# ============================================================
def compute_scaled_distances(df_candidates: pd.DataFrame,
                             target_row: pd.Series,
                             key_cols: list[str]) -> np.ndarray:

    key_cols = [c for c in key_cols if c in df_candidates.columns and c in target_row.index]
    if not key_cols:
        raise RuntimeError("❌ Nessuna colonna chiave per distanza soft!")

    X = df_candidates[key_cols].astype(float).values
    t = target_row[key_cols].astype(float).values

    # NaN → media colonna
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    X = np.where(np.isnan(X), col_means, X)
    t = np.where(np.isnan(t), col_means, t)

    # Normalizzazione
    col_std = np.nanstd(X, axis=0)
    col_std = np.where(col_std < 1e-6, 1.0, col_std)

    Xz = (X - col_means) / col_std
    tz = (t - col_means) / col_std

    diff = Xz - tz
    return np.sqrt(np.sum(diff * diff, axis=1))


# ============================================================
# 2. PESI (KERNEL ESPONENZIALE)
# ============================================================
def distances_to_weights(d: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    if len(d) == 0:
        return np.array([])

    if np.all(d == 0):
        return np.ones_like(d)

    p90 = np.percentile(d, 90)
    if p90 <= 0:
        d_scaled = d
    else:
        d_scaled = d / p90

    return np.exp(-alpha * d_scaled)


# ============================================================
# 3. ENGINE PRINCIPALE — API
# ============================================================
def run_soft_engine_api(
    target_row=None,
    target_match_id=None,
    slim=None,
    wide=None,
    top_n=80,
    min_neighbors=30,
):
    """
    Soft Engine API V2
    - Se target_match_id è presente → cerca la riga nel SLIM index
    - Se target_row è presente → usa quello (fixture future)
    """

    if slim is None or wide is None:
        raise RuntimeError("slim e wide index richiesti")

    # =====================================================
    # 1) Determina la riga target
    # =====================================================
    if target_match_id is not None:
        rows = slim.loc[slim["match_id"] == target_match_id]
        if rows.empty:
            return {"status": "error", "reason": "match_not_found_in_slim"}
        t0 = rows.iloc[0]
    else:
        if target_row is None:
            return {"status": "error", "reason": "missing_target_row"}
        t0 = target_row

    # estrai cluster e feature principali
    c1  = t0.get("cluster_1x2", np.nan)
    c25 = t0.get("cluster_ou25", np.nan)
    c15 = t0.get("cluster_ou15", np.nan)

    elo_t = t0.get("elo_diff", np.nan)
    lam_t = t0.get("lambda_total_form", np.nan)
    ms_t  = t0.get("market_sharpness", np.nan)

    # =====================================================
    # 2) FILTRI HARD PROGRESSIVI
    # =====================================================
    attempts = [
        dict(use_ou=True, widen=1.0, use_ms=True),
        dict(use_ou=True, widen=1.5, use_ms=True),
        dict(use_ou=True, widen=2.0, use_ms=True),
        dict(use_ou=False, widen=1.5, use_ms=True),
        dict(use_ou=False, widen=2.0, use_ms=False),
        dict(use_ou=False, widen=3.0, use_ms=False),
    ]

    def apply_filters(cfg):
        mask = (slim["cluster_1x2"] == c1)

        if cfg["use_ou"]:
            if "cluster_ou25" in slim.columns:
                mask &= (slim["cluster_ou25"] == c25)
            if "cluster_ou15" in slim.columns:
                mask &= (slim["cluster_ou15"] == c15)

        if not pd.isna(elo_t):
            d = 25 * cfg["widen"]
            mask &= slim["elo_diff"].between(elo_t - d, elo_t + d)

        if not pd.isna(lam_t):
            d = 0.40 * cfg["widen"]
            mask &= slim["lambda_total_form"].between(lam_t - d, lam_t + d)

        if cfg["use_ms"] and not pd.isna(ms_t):
            d = 0.04 * cfg["widen"]
            mask &= slim["market_sharpness"].between(ms_t - d, ms_t + d)

        cands = slim[mask].copy()
        if target_match_id is not None:
            cands = cands[cands["match_id"] != target_match_id]
        return cands

    candidates = None
    chosen_cfg = None

    for cfg in attempts:
        c = apply_filters(cfg)
        if len(c) >= min_neighbors:
            candidates = c
            chosen_cfg = cfg
            break

    if candidates is None:
        candidates = slim[slim["cluster_1x2"] == c1].copy()
        if target_match_id is not None:
            candidates = candidates[candidates["match_id"] != target_match_id]
        chosen_cfg = {"use_ou": False, "widen": 999, "use_ms": False}

    # =====================================================
    # 3) DISTANZA SOFT
    # =====================================================
    key_cols = [
        "elo_diff",
        "lambda_total_form",
        "tightness_index",           # 👈 NEW: entra nella distanza soft

        # se in futuro le rimetti nello SLIM, verranno usate automaticamente
        "market_sharpness",
        "delta_p1", "delta_p2",
        "delta_O25", "delta_U25",
    ]
    key_cols = [c for c in key_cols if c in slim.columns]

    dists = compute_scaled_distances(candidates, t0, key_cols)
    print('Dist', dists)
    candidates["distance_soft"] = dists

    candidates = candidates.sort_values("distance_soft").head(top_n).copy()

    # -----------------------------------------------------
    # PESI
    weights = distances_to_weights(candidates["distance_soft"].values)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    candidates["weight"] = weights

    # =====================================================
    # 4) UNIONE CON WIDE PER OUTCOME REALI
    # =====================================================
    wide_aff = wide.merge(
        candidates[["match_id", "weight", "distance_soft"]],
        on="match_id",
        how="inner",
    )

    gh = wide_aff["home_ft"].values.astype(float)
    ga = wide_aff["away_ft"].values.astype(float)
    w  = wide_aff["weight"].values.astype(float)
    W  = w.sum()

    p1  = (w * (gh > ga)).sum() / W
    px  = (w * (gh == ga)).sum() / W
    p2  = (w * (gh < ga)).sum() / W

    pO15 = (w * ((gh + ga) >= 2)).sum() / W
    pU15 = 1 - pO15

    pO25 = (w * ((gh + ga) >= 3)).sum() / W
    pU25 = 1 - pO25

    # OUTPUT
    return {
        "status": "ok",
        "clusters": {
            "cluster_1x2": int(c1) if not pd.isna(c1) else None,
            "cluster_ou25": int(c25) if not pd.isna(c25) else None,
            "cluster_ou15": int(c15) if not pd.isna(c15) else None,
        },
        "soft_probs": {
            "p1": p1, "px": px, "p2": p2,
            "pO15": pO15, "pU15": pU15,
            "pO25": pO25, "pU25": pU25,
        },
        "affini_stats": {
            "n_affini_soft": len(candidates),
            "avg_distance": float(candidates["distance_soft"].mean()),
        },
        "affini_list": candidates.merge(
            wide[["match_id", "home_ft", "away_ft"]],
            on="match_id",
            how="left",
        ).to_dict(orient="records"),
        "config_used": chosen_cfg,
    }