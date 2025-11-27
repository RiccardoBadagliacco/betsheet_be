#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
soft_engine_api_v2.py — Soft Engine V2 potenziato (con OU soft)

Funzione principale:
    run_soft_engine_api(
        target_row=None,
        target_match_id=None,
        slim=None,
        wide=None,
        top_n=80,
        min_neighbors=30,
    )

- Se target_match_id è passato → cerca nel SLIM index.
- Se target_row è passato → usa la riga (per fixture future).
- Usa:
    * cluster_1x2 (sempre)
    * cluster_ou25 / cluster_ou15 con tolleranza (distanza cluster <= k)
    * elo_diff, lambda_total_form, season_recency, tightness_index
    * eventuali delta_* / entropy_* / market_sharpness se presenti
- Restituisce:
    {
      status,
      clusters,
      soft_probs,
      affini_stats,
      affini_list,
      config_used
    }
"""

import numpy as np
import pandas as pd
from typing import Optional
from app.ml.correlazioni_affini_v2.common.betting_rules.index import evaluate_all_rules

# ============================================================
# 1. DISTANZA BLOCK-WEIGHTED
# ============================================================
def compute_block_weighted_distances(
    df_candidates: pd.DataFrame,
    target_row: pd.Series,
) -> np.ndarray:
    """
    Distanza soft potenziata a blocchi, con pesi diversi per:
      - struttura tecnica
      - market vs tecnico (delta)
      - shape del mercato
      - meta / tempo

    Usa solo le colonne effettivamente presenti in df_candidates & target_row.
    Se un blocco non ha colonne disponibili viene ignorato.
    """

    BLOCKS = {
        "structure": {
            "cols": [
                "elo_diff",
                "lambda_total_form",
                "lambda_total_market_ou25",
                "goal_supremacy_form",
                "goal_supremacy_market_ou25",
            ],
            "weight": 1.0,
        },
        "market_vs_tech": {
            "cols": [
                "delta_p1", "delta_p2",
                "delta_O25", "delta_U25",
                "delta_1x2_abs_sum",
                "delta_ou25_market_vs_form",
            ],
            "weight": 1.3,
        },
        "shape": {
            "cols": [
                "tightness_index",
                "entropy_bk_1x2", "entropy_pic_1x2",
                "entropy_bk_ou25", "entropy_pic_ou25",
            ],
            "weight": 0.8,
        },
        "meta": {
            "cols": [
                "season_recency",
            ],
            "weight": 0.5,
        },
    }

    n = len(df_candidates)
    if n == 0:
        return np.array([], dtype=float)

    dist_total = np.zeros(n, dtype=float)
    used_any_block = False

    for block_name, block in BLOCKS.items():
        cols = [
            c for c in block["cols"]
            if c in df_candidates.columns and c in target_row.index
        ]
        if not cols:
            continue

        used_any_block = True
        w = float(block["weight"])

        X = df_candidates[cols].astype(float).values
        t = target_row[cols].astype(float).values

        # imputazione NaN
        col_means = np.nanmean(X, axis=0)
        # 👉 se TUTTO è NaN per questo blocco → lo skippiamo
        if np.isnan(col_means).all():
            continue
        
        col_means = np.where(np.isnan(col_means), 0.0, col_means)

        X = np.where(np.isnan(X), col_means, X)
        t = np.where(np.isnan(t), col_means, t)

        col_std = np.nanstd(X, axis=0)
        col_std = np.where(col_std < 1e-6, 1.0, col_std)

        Xz = (X - col_means) / col_std
        tz = (t - col_means) / col_std

        diff = Xz - tz
        dist_block = np.sqrt(np.sum(diff * diff, axis=1))

        # accumula blocco con peso
        dist_total += (w * dist_block) ** 2

    if not used_any_block:
        # fallback: distanza nulla (tutti equivalenti)
        return np.zeros(n, dtype=float)

    return np.sqrt(dist_total)


# ============================================================
# 2. PESI (KERNEL ESPONENZIALE)
# ============================================================
def distances_to_weights(d: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """
    Converte distanze in pesi tramite kernel esponenziale.
    - scala le distanze col percentile 90 per robustezza
    """
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return d

    if np.all(d == 0):
        return np.ones_like(d)

    p90 = np.percentile(d, 90)
    if p90 <= 0:
        d_scaled = d
    else:
        d_scaled = d / p90

    w = np.exp(-alpha * d_scaled)
    return w


# ============================================================
# 3. ENGINE PRINCIPALE — API
# ============================================================
def run_soft_engine_api(
    target_row: Optional[pd.Series] = None,
    target_match_id: Optional[str] = None,
    slim: Optional[pd.DataFrame] = None,
    wide: Optional[pd.DataFrame] = None,
    top_n: int = 80,
    min_neighbors: int = 30,
):
    """
    Soft Engine V2 (potenziato, block-weighted + OU soft).

    Parametri
    ---------
    target_row : Series opzionale
        Riga "tipo slim" per fixture future (stesse colonne dello SLIM).
    target_match_id : str opzionale
        match_id già presente nello SLIM.
    slim : DataFrame
        Indice SLIM (step4b_affini_index_slim_v2.parquet).
    wide : DataFrame
        Indice WIDE (step4a_affini_index_wide_v2.parquet).
    top_n : int
        Numero massimo di affini soft usati per le probabilità.
    min_neighbors : int
        Minimo di affini richiesti dopo hard-filter.
    """

    if slim is None or wide is None:
        raise RuntimeError("❌ run_soft_engine_api richiede 'slim' e 'wide' non nulli")

    # =====================================================
    # 1) Determina la riga target
    # =====================================================
    if target_match_id is not None:
        rows = slim.loc[slim["match_id"] == target_match_id]
        if rows.empty:
            return {
                "status": "error",
                "reason": "match_not_found_in_slim",
                "target_match_id": target_match_id,
            }
        t0 = rows.iloc[0]
    else:
        if target_row is None:
            return {
                "status": "error",
                "reason": "missing_target_row",
            }
        t0 = target_row

    # =====================================================
    # 2) Estrai cluster e driver principali
    # =====================================================
    c1  = t0.get("cluster_1x2", np.nan)
    c25 = t0.get("cluster_ou25", np.nan)
    c15 = t0.get("cluster_ou15", np.nan)

    elo_t = t0.get("elo_diff", np.nan)
    lam_t = t0.get("lambda_total_form", np.nan)
    rec_t = t0.get("season_recency", np.nan)
    ms_t  = t0.get("market_sharpness", np.nan) if "market_sharpness" in t0.index else np.nan

    # cluster 1X2 è obbligatorio per logica affini
    if pd.isna(c1):
        return {
            "status": "error",
            "reason": "missing_cluster_1x2",
        }

    # =====================================================
    # 3) HARD FILTER PROGRESSIVO (con OU soft)
    # =====================================================
    # Ogni config allenta progressivamente i vincoli
    attempts = [
        # 1) Molto stretto: OU identici, finestre standard
        dict(
            name="strict_all",
            use_ou=True,
            ou25_tol=0,
            ou15_tol=0,
            use_rec=True,
            use_ms=True,
            elo_w=1.0,
            lam_w=1.0,
            rec_w=1.0,
        ),
        # 2) OU +/- 1, recency meno importante
        dict(
            name="ou_tol1",
            use_ou=True,
            ou25_tol=1,
            ou15_tol=1,
            use_rec=False,
            use_ms=True,
            elo_w=1.0,
            lam_w=1.2,
            rec_w=1.0,
        ),
        # 3) OU +/- 2, elo/lambda più larghi
        dict(
            name="ou_tol2_wider_elo_lam",
            use_ou=True,
            ou25_tol=2,
            ou15_tol=2,
            use_rec=False,
            use_ms=True,
            elo_w=1.5,
            lam_w=1.5,
            rec_w=1.0,
        ),
        # 4) senza OU, mantengo ms + elo/lambda larghi
        dict(
            name="no_ou_keep_ms",
            use_ou=False,
            ou25_tol=None,
            ou15_tol=None,
            use_rec=False,
            use_ms=True,
            elo_w=1.5,
            lam_w=1.5,
            rec_w=1.2,
        ),
        # 5) solo cluster 1x2 + elo/lambda molto larghi
        dict(
            name="no_ou_no_ms",
            use_ou=False,
            ou25_tol=None,
            ou15_tol=None,
            use_rec=False,
            use_ms=False,
            elo_w=2.0,
            lam_w=2.0,
            rec_w=1.5,
        ),
    ]

    # finestre base (prima di moltiplicare per *w)
    ELO_BASE = 25.0
    LAM_BASE = 0.40
    REC_BASE = 0.15
    MS_BASE  = 0.04

    def apply_filters(cfg):
        mask = (slim["cluster_1x2"] == c1)

        # --- OU SOFT ---------------------------------------
        if cfg["use_ou"]:
            # OU2.5
            if "cluster_ou25" in slim.columns and not pd.isna(c25):
                tol25 = cfg.get("ou25_tol", 0)
                if tol25 is None or tol25 <= 0:
                    mask &= (slim["cluster_ou25"] == c25)
                else:
                    mask &= slim["cluster_ou25"].between(c25 - tol25, c25 + tol25)
            # OU1.5
            if "cluster_ou15" in slim.columns and not pd.isna(c15):
                tol15 = cfg.get("ou15_tol", 0)
                if tol15 is None or tol15 <= 0:
                    mask &= (slim["cluster_ou15"] == c15)
                else:
                    mask &= slim["cluster_ou15"].between(c15 - tol15, c15 + tol15)

        # elo_diff
        if not pd.isna(elo_t) and "elo_diff" in slim.columns:
            d = ELO_BASE * cfg["elo_w"]
            mask &= slim["elo_diff"].between(elo_t - d, elo_t + d)

        # lambda_total_form
        if not pd.isna(lam_t) and "lambda_total_form" in slim.columns:
            d = LAM_BASE * cfg["lam_w"]
            mask &= slim["lambda_total_form"].between(lam_t - d, lam_t + d)

        # season_recency
        if cfg["use_rec"] and (not pd.isna(rec_t)) and "season_recency" in slim.columns:
            d = REC_BASE * cfg["rec_w"]
            mask &= slim["season_recency"].between(rec_t - d, rec_t + d)

        # market_sharpness (solo se esiste nel SLIM)
        if cfg["use_ms"] and (not pd.isna(ms_t)) and "market_sharpness" in slim.columns:
            d = MS_BASE
            mask &= slim["market_sharpness"].between(ms_t - d, ms_t + d)

        cands = slim[mask].copy()

        # escludi il match stesso, se storico
        if target_match_id is not None and "match_id" in cands.columns:
            cands = cands[cands["match_id"] != target_match_id]

        return cands

    candidates = None
    chosen_cfg = None

    for cfg in attempts:
        cands = apply_filters(cfg)
        if len(cands) >= min_neighbors:
            candidates = cands
            chosen_cfg = cfg
            break

    # fallback: tutti i match dello stesso cluster 1x2
    if candidates is None:
        mask = (slim["cluster_1x2"] == c1)
        candidates = slim[mask].copy()
        if target_match_id is not None and "match_id" in candidates.columns:
            candidates = candidates[candidates["match_id"] != target_match_id]
        chosen_cfg = {
            "name": "fallback_cluster_only",
            "use_ou": False,
            "ou25_tol": None,
            "ou15_tol": None,
            "use_rec": False,
            "use_ms": False,
            "elo_w": None,
            "lam_w": None,
            "rec_w": None,
        }

    if candidates.empty:
        return {
            "status": "error",
            "reason": "no_candidates_found",
            "config_used": chosen_cfg,
        }

    # =====================================================
    # 4) DISTANZA SOFT + PESI
    # =====================================================
    dists = compute_block_weighted_distances(candidates, t0)
    candidates = candidates.copy()
    candidates["distance_soft"] = dists

    # ordina per distanza e prendi top_n
    candidates = candidates.sort_values("distance_soft", ascending=True).head(top_n)

    weights = distances_to_weights(candidates["distance_soft"].values)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    candidates["weight"] = weights

    # =====================================================
    # 5) UNIONE CON WIDE (per risultati reali)
    # =====================================================
    if "match_id" not in candidates.columns:
        return {
            "status": "error",
            "reason": "candidates_missing_match_id",
        }

    wide_aff = wide.merge(
        candidates[["match_id", "weight", "distance_soft"]],
        on="match_id",
        how="inner",
    )

    if wide_aff.empty or "home_ft" not in wide_aff.columns or "away_ft" not in wide_aff.columns:
        return {
            "status": "error",
            "reason": "wide_missing_results_or_empty",
        }

    gh = wide_aff["home_ft"].values.astype(float)
    ga = wide_aff["away_ft"].values.astype(float)
    w  = wide_aff["weight"].values.astype(float)
    W  = w.sum()

    if W <= 0:
        W = 1.0

    # 1X2
    p1 = (w * (gh > ga)).sum() / W
    px = (w * (gh == ga)).sum() / W
    p2 = (w * (gh < ga)).sum() / W

    # O/U 1.5
    goals = gh + ga
    pO15 = (w * (goals >= 2)).sum() / W
    pU15 = 1.0 - pO15

    # O/U 2.5
    pO25 = (w * (goals >= 3)).sum() / W
    pU25 = 1.0 - pO25

    # =====================================================
    # 6) COSTRUZIONE OUTPUT
    # =====================================================
    clusters_out = {
        "cluster_1x2": int(c1) if not pd.isna(c1) else None,
        "cluster_ou25": int(c25) if (not pd.isna(c25)) else None,
        "cluster_ou15": int(c15) if (not pd.isna(c15)) else None,
    }

    soft_probs = {
        "p1": float(p1),
        "px": float(px),
        "p2": float(p2),
        "pO15": float(pO15),
        "pU15": float(pU15),
        "pO25": float(pO25),
        "pU25": float(pU25),
    }

    affini_stats = {
        "n_affini_soft": int(len(candidates)),
        "avg_distance": float(candidates["distance_soft"].mean()),
        "min_distance": float(candidates["distance_soft"].min()),
        "max_distance": float(candidates["distance_soft"].max()),
    }

    # affini_list: unione candidati + risultato finale
    affini_df = candidates.merge(
        wide[["match_id", "home_ft", "away_ft"]],
        on="match_id",
        how="left",
    )

    affini_list = affini_df.to_dict(orient="records")

    target_date = pd.to_datetime(t0.get("date"), errors="coerce")

    # Usa solo le partite della stessa season del match target (se disponibile)
    season = t0.get("season")
    if pd.notna(season) and "season" in wide.columns:
        history_base = wide[wide["season"] == season]
    else:
        history_base = wide

    history = history_base[["match_id", "date", "home_team", "away_team", "home_ft", "away_ft"]].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")

    # Tieni solo match precedenti alla data target (se disponibile)
    if pd.notna(target_date):
        history = history[(history["date"].notna()) & (history["date"] < target_date)].copy()

    # Costruisco dizionario {team_name: [match_records]}
    team_history = {}

    for _, r in history.iterrows():
        # considera solo partite con risultato completo
        if pd.isna(r["home_ft"]) or pd.isna(r["away_ft"]):
            continue

        # home side
        team_history.setdefault(r["home_team"], []).append({
            "match_id": r["match_id"],
            "date": r["date"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_ft": int(r["home_ft"]),
            "away_ft": int(r["away_ft"]),
        })
        # away side
        team_history.setdefault(r["away_team"], []).append({
            "match_id": r["match_id"],
            "date": r["date"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_ft": int(r["home_ft"]),
            "away_ft": int(r["away_ft"]),
        })

    # Ordino le liste per data
    for t in team_history:
        team_history[t].sort(key=lambda x: x["date"])

    home_team = t0.get("home_team")
    if pd.notna(home_team) and home_team in team_history:
        print(f"\nTutte le partite di {home_team} (stessa season):")
        for m in team_history[home_team]:
            d = m["date"].strftime("%Y-%m-%d") if pd.notna(m["date"]) else "N/A"
            hf = m["home_ft"] if m["home_ft"] is not None else "-"
            af = m["away_ft"] if m["away_ft"] is not None else "-"
            print(f"  {d} | {m['home_team']} {hf} - {af} {m['away_team']}")
        print("")
    else:
        print(f"\nNessuna history disponibile per il team di casa: {home_team}\n")

    away_team = t0.get("away_team")
    if pd.notna(away_team) and away_team in team_history:
        print(f"Tutte le partite di {away_team} (stessa season):")
        for m in team_history[away_team]:
            d = m["date"].strftime("%Y-%m-%d") if pd.notna(m["date"]) else "N/A"
            hf = m["home_ft"] if m["home_ft"] is not None else "-"
            af = m["away_ft"] if m["away_ft"] is not None else "-"
            print(f"  {d} | {m['home_team']} {hf} - {af} {m['away_team']}")
        print("")
    else:
        print(f"Nessuna history disponibile per il team ospite: {away_team}\n")

    ctx = {
        "clusters": clusters_out,
        "soft_probs": soft_probs,
        "team_history": team_history,
        # in futuro puoi aggiungere altre info di contesto qui
    }

    ML_COLS = [
        "home_form_matches_lastN",
        "home_form_gf_avg_lastN",
        "home_form_ga_avg_lastN",
        "away_form_matches_lastN",
        "away_form_gf_avg_lastN",
        "away_form_ga_avg_lastN"
    ]

    # Se il target è storico ed esiste nel WIDE → merge diretto
    if "match_id" in t0 and "match_id" in wide.columns:
        wid = wide[wide["match_id"] == t0["match_id"]]

        if len(wid) > 0:
            w0 = wid.iloc[0]
            for col in ML_COLS:
                if col in w0.index:
                    t0[col] = w0[col]
                else:
                    t0[col] = np.nan
        else:
            # Match futuro → assegna NaN
            for col in ML_COLS:
                t0[col] = np.nan
    else:
        # Caso generico → assegna NaN
        for col in ML_COLS:
            t0[col] = np.nan

    return {
        "status": "ok",
        "clusters": clusters_out,
        "soft_probs": soft_probs,
        "affini_stats": affini_stats,
        "affini_list": affini_list,
        "config_used": chosen_cfg,
        "alerts": evaluate_all_rules(t0, ctx)
    }
