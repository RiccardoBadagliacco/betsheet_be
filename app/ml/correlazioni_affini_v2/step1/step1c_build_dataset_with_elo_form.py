# app/ml/correlazioni_affini_v2/step1/step1c_build_dataset_with_elo_form.py
"""
STEP1C v4 — Merge base + Elo + Form + Feature Engineering MASTER (estesa)

Input:
    - step0_dataset_base.parquet
    - step1a_elo.parquet
    - step1b_form_recent.parquet

Output:
    - step1c_dataset_with_elo_form.parquet

In questo step costruiamo il dataset "master" di training con:
    - info base match
    - quote e probabilità bookmaker 1X2 + OU2.5
    - Elo pre/post + expected score
    - Form recente home/away (rolling N match)
    - Feature derivate (team strength, mismatch Elo vs mercato, etc.)
    - NUOVE FEATURE v4:
        * Goal model da OU2.5 (lambda impliciti via Poisson)
        * Delta mercato vs forma sul totale gol
        * Supremazia gol (market, form, reale) + errori
        * Momentum / fatica (rest diff & short rest)
"""

from __future__ import annotations
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
from math import exp, factorial


# ============================================
# PATH
# ============================================

BASE_DIR = Path(__file__).resolve().parents[2]  # .../app/ml
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"

PATH_STEP0 = DATA_DIR / "step0_dataset_base.parquet"
PATH_STEP1A = DATA_DIR / "step1a_elo.parquet"
PATH_STEP1B = DATA_DIR / "step1b_form_recent.parquet"

OUT_PATH = DATA_DIR / "step1c_dataset_with_elo_form.parquet"
PATH_STEP1Z = DATA_DIR / "step1z_fixture_features.parquet"


# ============================================
# UTILS POISSON (per OU2.5 / goal model)
# ============================================

def _poisson_p_leq(k: int, lam: float) -> float:
    """
    P(X <= k) per X ~ Poisson(lam)
    """
    if lam <= 0:
        return 0.0
    s = 0.0
    for i in range(k + 1):
        s += exp(-lam) * lam**i / factorial(i)
    return s


def _poisson_p_over25(lam: float) -> float:
    """
    P(total_goals >= 3) per Poisson(lam)
    """
    return 1.0 - _poisson_p_leq(2, lam)


def implied_lambda_from_p_over25(p_over25: float,
                                 tol: float = 1e-4,
                                 max_iter: int = 50) -> Optional[float]:
    """
    Data una probabilità di OVER 2.5 (0..1), ricava lambda totale
    di un goal model Poisson tale che P(X>=3) ≈ p_over25.

    Usiamo una bisezione semplice su [lam_low, lam_high].
    """
    if p_over25 is None or np.isnan(p_over25):
        return np.nan
    p_over25 = float(p_over25)
    if p_over25 <= 0.0:
        return 0.0
    if p_over25 >= 1.0:
        return 10.0  # limite superiore “assurdo” ma safe

    lam_low, lam_high = 0.01, 8.0  # range ragionevole per il calcio

    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_low + lam_high)
        p_mid = _poisson_p_over25(lam_mid)
        if abs(p_mid - p_over25) < tol:
            return lam_mid
        if p_mid < p_over25:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    return lam_mid


# ============================================
# FEATURE ENGINEERING BLOCCO ELO/MERCATO/FORM
# (quello che avevamo in v3 + estensioni v4)
# ============================================

def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature derivate da Elo + mercato 1X2.
    """
    # Somma & share
    df["elo_sum"] = df["elo_home_pre"] + df["elo_away_pre"]
    df["elo_home_share"] = df["elo_home_pre"] / df["elo_sum"]
    df["elo_away_share"] = df["elo_away_pre"] / df["elo_sum"]

    # Probabilità “tecniche” da Elo (già abbiamo exp_home / exp_away)
    df["elo_prob_home"] = df["exp_home"]
    df["elo_prob_away"] = df["exp_away"]

    # Delta vs bookmaker
    df["elo_delta_prob_home_vs_bk"] = df["elo_prob_home"] - df["bk_p1"]
    df["elo_delta_prob_away_vs_bk"] = df["elo_prob_away"] - df["bk_p2"]

    df["elo_market_misalignment_1x2"] = (
        df["elo_delta_prob_home_vs_bk"].abs()
        + df["elo_delta_prob_away_vs_bk"].abs()
    )

    return df


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature derivate dalla forma recente.
    """
    df["form_pts_diff"] = (
        df["home_form_pts_avg_lastN"] - df["away_form_pts_avg_lastN"]
    )
    df["form_gf_diff"] = (
        df["home_form_gf_avg_lastN"] - df["away_form_gf_avg_lastN"]
    )
    df["form_ga_diff"] = (
        df["home_form_ga_avg_lastN"] - df["away_form_ga_avg_lastN"]
    )
    df["form_win_rate_diff"] = (
        df["home_form_win_rate_lastN"] - df["away_form_win_rate_lastN"]
    )
    df["form_matches_diff"] = (
        df["home_form_matches_lastN"] - df["away_form_matches_lastN"]
    )

    # Indici attacco-difesa
    df["home_form_att_index"] = (
        df["home_form_gf_avg_lastN"] - df["home_form_ga_avg_lastN"]
    )
    df["away_form_att_index"] = (
        df["away_form_gf_avg_lastN"] - df["away_form_ga_avg_lastN"]
    )
    df["form_att_diff"] = df["home_form_att_index"] - df["away_form_att_index"]

    return df


def add_team_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Team strength sintetico (Elo + Form).
    Non è un valore assoluto, ma un indice comparativo.
    """
    # Piccolo scaling dell’Elo per non avere numeri enormi
    # e combinazione con la forma (punti + indice attacco).
    df["team_strength_home"] = (
        (df["elo_home_pre"] - 1500.0) / 40.0
        + df["home_form_pts_avg_lastN"]
        + 0.5 * df["home_form_att_index"]
    )
    df["team_strength_away"] = (
        (df["elo_away_pre"] - 1500.0) / 40.0
        + df["away_form_pts_avg_lastN"]
        + 0.5 * df["away_form_att_index"]
    )
    df["team_strength_diff"] = df["team_strength_home"] - df["team_strength_away"]

    return df


def add_market_features_1x2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature di struttura del mercato 1X2 (entropia & gap).
    Richiede:
        - bk_p1, bk_px, bk_p2
        - entropy_bk_1x2 (già calcolata in step0)
        - fav_side_1x2, fav_prob_1x2
    """
    # Secondo favorito: max(bk_px, min(bk_p1, bk_p2))
    probs = df[["bk_p1", "bk_px", "bk_p2"]].values
    # ordiniamo per riga
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]  # discendente
    df["second_fav_prob_1x2"] = sorted_probs[:, 1]
    df["fav_prob_gap_1x2"] = sorted_probs[:, 0] - sorted_probs[:, 1]

    # Spread home-away
    df["spread_home_away_probs_1x2"] = df["bk_p1"] - df["bk_p2"]

    # Market balance index = quanto il mercato è “sbilanciato” vs uniforme
    # (1 - entropia) in pratica, ma usiamo già entropy_bk_1x2
    df["market_balance_index_1x2"] = 1.0 - df["entropy_bk_1x2"]

    # Segno del mercato rispetto alla forza tecnica (Elo)
    # >0: mercato va nella stessa direzione di Elo (home forte, bk_p1 > bk_p2)
    # <0: mercato controtendenza
    df["market_sign_vs_strength"] = np.sign(
        df["spread_home_away_probs_1x2"] * df["elo_diff"]
    )

    return df


def add_market_features_ou25(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature sugli over/under 2.5 basate su bk_pO25/bk_pU25.
    """
    df["ou25_has_market"] = (~df["bk_pO25"].isna()).astype(int)

    # Bias semplice: preferenza del mercato per l'over vs 50/50
    df["ou25_bias_over"] = df["bk_pO25"] - 0.5

    # Balance OU: quanto il mercato è vicino al 50/50
    df["ou25_balance_index"] = (df["bk_pO25"] - df["bk_pU25"]).abs()

    return df


def add_season_and_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Season recency + feature di calendario (rest & density).
    Richiede che 'date' sia datetime.
    """
    # Season recency normalizzata per ogni stagione (0..1 in base alla data dentro la stagione)
    def _season_recency(group: pd.DataFrame) -> pd.Series:
        # Assumiamo date ordinate
        t = group["date"].values.astype("datetime64[D]").astype(float)
        t_min = t.min()
        t_max = t.max()
        if t_max == t_min:
            return pd.Series(0.5, index=group.index)
        return pd.Series((t - t_min) / (t_max - t_min), index=group.index)

    df["season_recency"] = (
        df.groupby("season", group_keys=False)
        .apply(_season_recency)
        .reindex(df.index)
    )

    # Days since last match (home & away) e match density
    df = df.sort_values("date").reset_index(drop=True)

    # helper: calcola days since last per team
    def _compute_days_since_last(team_col: str, out_col: str) -> pd.DataFrame:
        last_date_per_team = {}
        days = []
        for _, r in df.iterrows():
            team = r[team_col]
            d = r["date"]
            prev_date = last_date_per_team.get(team)
            if prev_date is None:
                days.append(np.nan)
            else:
                days.append((d - prev_date).days)
            last_date_per_team[team] = d
        df[out_col] = days
        return df

    df = _compute_days_since_last("home_team", "days_since_last_home")
    df = _compute_days_since_last("away_team", "days_since_last_away")

    # match density = numero di match giocati negli ultimi 30 giorni (home+away)
    window_days = 30

    # Costruisco una tabella "long" team-date
    tmp_home = df[["match_id", "date", "home_team"]].rename(
        columns={"home_team": "team"}
    )
    tmp_away = df[["match_id", "date", "away_team"]].rename(
        columns={"away_team": "team"}
    )
    tmp_all = pd.concat([tmp_home, tmp_away], ignore_index=True)
    tmp_all = tmp_all.sort_values(["team", "date"])

    # per ogni riga, conta quanti match dello stesso team negli ultimi 30 giorni
    match_density = []
    last_seen = {}

    for _, r in tmp_all.iterrows():
        team = r["team"]
        date = r["date"]
        if team not in last_seen:
            last_seen[team] = []
        # rimuovi match più vecchi di window_days
        last_seen[team] = [d for d in last_seen[team] if (date - d).days <= window_days]
        # densità = match negli ultimi window_days (senza contare quello di oggi)
        match_density.append(len(last_seen[team]))
        last_seen[team].append(date)

    tmp_all["match_density"] = match_density

    # Ora riportiamo la densità al livello match_id, come media tra home e away
    home_density = tmp_home.merge(
        tmp_all[["match_id", "team", "match_density"]],
        on=["match_id", "team"],
        how="left",
    ).rename(columns={"match_density": "match_density_home"})
    away_density = tmp_away.merge(
        tmp_all[["match_id", "team", "match_density"]],
        on=["match_id", "team"],
        how="left",
    ).rename(columns={"match_density": "match_density_away"})

    dens = home_density[["match_id", "match_density_home"]].merge(
        away_density[["match_id", "match_density_away"]],
        on="match_id",
        how="outer",
    )

    df = df.merge(dens, on="match_id", how="left")
    df["match_density_index"] = (
        df[["match_density_home", "match_density_away"]].mean(axis=1) / 10.0
    )

    return df


# ============================================
# NUOVE FEATURE v4 — GOAL MODEL & FATICA
# ============================================

def add_goal_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Goal model misto:
        - lambda totale implicito da OU2.5 mercato (Poisson inverso)
        - lambda form-based da gf medi
        - supremazia mercato vs forma vs reale
    Tutto opzionale: se OU2.5 manca, le colonne risultano NaN.
    """

    # --- lambda totale implicito dal mercato OU2.5 (solo partite con bk_pO25 valido)
    lam_market = []
    for p in df["bk_pO25"]:
        if pd.isna(p):
            lam_market.append(np.nan)
        else:
            lam_market.append(implied_lambda_from_p_over25(p))
    df["lambda_total_market_ou25"] = lam_market

    # --- Decomposizione home/away basata su Elo_diff
    # share_home ≈ 0.5 + (elo_diff / 400) / 2, poi clippato
    share_home = 0.5 + (df["elo_diff"] / 400.0) / 2.0
    share_home = share_home.clip(0.1, 0.9)
    share_away = 1.0 - share_home

    df["lambda_home_market_ou25"] = df["lambda_total_market_ou25"] * share_home
    df["lambda_away_market_ou25"] = df["lambda_total_market_ou25"] * share_away

    # --- lambda da forma (gf medi)
    # usiamo gf_avg_lastN come proxy di lambda; evitiamo 0 assoluti
    eps = 0.05
    df["lambda_home_form"] = df["home_form_gf_avg_lastN"].clip(lower=eps)
    df["lambda_away_form"] = df["away_form_gf_avg_lastN"].clip(lower=eps)
    df["lambda_total_form"] = df["lambda_home_form"] + df["lambda_away_form"]

    # Probabilità di over 2.5 da lambda_total_form
    def _p_over25_form(lam):
        if pd.isna(lam) or lam <= 0:
            return np.nan
        return _poisson_p_over25(lam)

    df["tech_pO25_form"] = df["lambda_total_form"].apply(_p_over25_form)

    # Delta mercato vs forma sul totale gol (over 2.5)
    df["delta_ou25_market_vs_form"] = df["bk_pO25"] - df["tech_pO25_form"]

    # --- Supremazia gol: market vs form vs reale
    df["goal_supremacy_market_ou25"] = (
        df["lambda_home_market_ou25"] - df["lambda_away_market_ou25"]
    )
    df["goal_supremacy_form"] = (
        df["lambda_home_form"] - df["lambda_away_form"]
    )
    df["goal_supremacy_real"] = df["home_ft"] - df["away_ft"]

    df["goal_supremacy_error_market"] = (
        df["goal_supremacy_market_ou25"] - df["goal_supremacy_real"]
    )
    df["goal_supremacy_error_form"] = (
        df["goal_supremacy_form"] - df["goal_supremacy_real"]
    )

    return df


def add_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature di fatica / momentum basate sul tempo di recupero.
    Richiede:
        - days_since_last_home
        - days_since_last_away
        - match_density_index (già calcolato)
    """
    # differenza di rest tra home e away
    df["rest_diff_days"] = df["days_since_last_home"] - df["days_since_last_away"]

    # short rest (<=3 giorni) — NaN -> 0
    df["short_rest_home"] = np.where(df["days_since_last_home"] <= 3, 1.0, 0.0)
    df["short_rest_away"] = np.where(df["days_since_last_away"] <= 3, 1.0, 0.0)
    df.loc[df["days_since_last_home"].isna(), "short_rest_home"] = 0.0
    df.loc[df["days_since_last_away"].isna(), "short_rest_away"] = 0.0

    # vantaggio di rest (>=2 giorni in più di pausa)
    df["rest_advantage_home"] = np.where(df["rest_diff_days"] >= 2, 1.0, 0.0)
    df["rest_advantage_away"] = np.where(df["rest_diff_days"] <= -2, 1.0, 0.0)
    df.loc[df["rest_diff_days"].isna(), ["rest_advantage_home", "rest_advantage_away"]] = 0.0

    return df


# ============================================
# MAIN
# ============================================

def main():
    print("============================================")
    print("🚀 STEP1C v4 — Merge base + Elo + Form + Feature Engineering MASTER")
    print("============================================")
    print(f"📥 STEP0  : {PATH_STEP0}")
    print(f"📥 STEP1A : {PATH_STEP1A}")
    print(f"📥 STEP1B : {PATH_STEP1B}")
    print(f"💾 Output : {OUT_PATH}")
    print("============================================")

    # -------------------------------
    # Carico i dataset
    # -------------------------------
    df0 = pd.read_parquet(PATH_STEP0)
    df_elo = pd.read_parquet(PATH_STEP1A)
    df_form = pd.read_parquet(PATH_STEP1B)

    df_fix = None
    if PATH_STEP1Z.exists():
        print(f"📥 STEP1Z trovato: {PATH_STEP1Z}")
        df_fix = pd.read_parquet(PATH_STEP1Z)
    else:
        print("⚠️ Nessun step1z_fixture_features trovato → ignorato.")

    print(f"📏 Shape STEP0  (base) : {df0.shape}")
    print(f"📏 Shape STEP1A (elo)  : {df_elo.shape}")
    print(f"📏 Shape STEP1B (form) : {df_form.shape}")

    # -------------------------------
    # Controlli rapidi sui match_id
    # -------------------------------
    set0 = set(df0["match_id"])
    setElo = set(df_elo["match_id"])
    setForm = set(df_form["match_id"])

    if set0 != setElo:
        print("⚠️ WARNING: STEP0 e STEP1A hanno set di match_id diversi!")
        print("   STEP0-only :", len(set0 - setElo))
        print("   STEP1A-only:", len(setElo - set0))
    if set0 != setForm:
        print("⚠️ WARNING: STEP0 e STEP1B hanno set di match_id diversi!")
        print("   STEP0-only :", len(set0 - setForm))
        print("   STEP1B-only:", len(setForm - set0))

    # -------------------------------
    # Riduzione colonne Elo / Form
    # -------------------------------
    elo_extra_cols = [
        "match_id",
        "elo_home_pre",
        "elo_away_pre",
        "exp_home",
        "exp_away",
        "elo_home_post",
        "elo_away_post",
        "elo_diff",
    ]
    elo_extra_cols = [c for c in elo_extra_cols if c in df_elo.columns]
    df_elo_red = df_elo[elo_extra_cols].copy()

    form_extra_cols = [
        "match_id",
        "home_form_matches_lastN",
        "home_form_pts_avg_lastN",
        "home_form_gf_avg_lastN",
        "home_form_ga_avg_lastN",
        "home_form_win_rate_lastN",
        "away_form_matches_lastN",
        "away_form_pts_avg_lastN",
        "away_form_gf_avg_lastN",
        "away_form_ga_avg_lastN",
        "away_form_win_rate_lastN",
    ]
    form_extra_cols = [c for c in form_extra_cols if c in df_form.columns]
    df_form_red = df_form[form_extra_cols].copy()

    print(f"📦 Colonne Elo  utilizzate : {elo_extra_cols}")
    print(f"📦 Colonne Form utilizzate : {form_extra_cols}")

    # -------------------------------
    # Merge base + Elo + Form
    # -------------------------------
    df = df0.merge(df_elo_red, on="match_id", how="left")
    print(f"📏 Dopo merge con Elo : {df.shape}")
    df = df.merge(df_form_red, on="match_id", how="left")
    print(f"📏 Dopo merge con Form: {df.shape}")

    # ======================================================
    # 🔵 MERGE FIXTURE FEATURE (step1z)
    #    → completa Elo + Form mancanti
    # ======================================================
    if df_fix is not None and not df_fix.empty:
        print("🔄 Merge FEATURE FIXTURE (step1z)…")

        before_missing = df["elo_home_pre"].isna().sum()
        print(f"   Elo mancanti prima del merge: {before_missing}")

        df = df.merge(df_fix, on="match_id", how="left", suffixes=("", "_fx"))

        # Sostituisci solo se le versioni "normali" sono NaN
        for col in ["elo_home_pre", "elo_away_pre", "elo_diff",
                    "lambda_home_form", "lambda_away_form", "lambda_total_form"]:
            fx_col = col + "_fx"
            if fx_col in df.columns:
                df[col] = df[col].fillna(df[fx_col])
                df = df.drop(columns=[fx_col])

        after_missing = df["elo_home_pre"].isna().sum()
        print(f"   Elo mancanti dopo merge: {after_missing}")


    # Assicuriamoci che date sia datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # -------------------------------
    # Feature engineering
    # -------------------------------
    print("⚙️  Costruzione feature ELO / mercato...")
    df = add_elo_features(df)

    print("⚙️  Costruzione feature Form...")
    df = add_form_features(df)

    print("⚙️  Costruzione feature Team Strength (Elo+Form)...")
    df = add_team_strength_features(df)

    print("⚙️  Costruzione feature mercato 1X2...")
    df = add_market_features_1x2(df)

    print("⚙️  Costruzione feature mercato OU 2.5...")
    df = add_market_features_ou25(df)

    print("⚙️  Aggiunta season_recency + calendario...")
    df = add_season_and_calendar_features(df)

    print("⚙️  [v4] Costruzione goal model (OU2.5 → lambda)...")
    df = add_goal_model_features(df)

    print("⚙️  [v4] Costruzione feature fatica / rest advantage...")
    df = add_fatigue_features(df)

    # ------------------------------------------------------
    # 🎯  TIGHTNESS INDEX (misura di coesione della partita)
    # ------------------------------------------------------

    # bk_pU25 può essere NaN per match senza OU 2.5 → riempiamo con mediana
    if "bk_pU25" in df.columns:
        u25 = df["bk_pU25"].astype(float)
        median_u25 = float(u25[u25.notna()].median()) if u25.notna().any() else 0.5
        u25_filled = u25.fillna(median_u25)
    else:
        u25_filled = 0.5

    # fav_prob_1x2 e entropy_bk_1x2 esistono già nella pipeline
    fav_prob = df["fav_prob_1x2"].astype(float)
    ent_1x2  = df["entropy_bk_1x2"].astype(float)

    # Tightness index: 0 (caotica) → 1 (rigida/pronosticabile)
    df["tightness_index"] = (
        0.4 * u25_filled +
        0.3 * (1.0 - fav_prob) +
        0.3 * ent_1x2
    )

    # Clamp [0,1] per sicurezza numerica
    df["tightness_index"] = df["tightness_index"].clip(0.0, 1.0)

    # Ordiniamo per data giusto per coerenza
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    print("============================================")
    print("📊 Anteprima colonne chiave aggiunte (v4):")
    cols_preview = [
        "bk_pO25",
        "bk_pU25",
        "lambda_total_market_ou25",
        "lambda_home_market_ou25",
        "lambda_away_market_ou25",
        "lambda_total_form",
        "tech_pO25_form",
        "delta_ou25_market_vs_form",
        "goal_supremacy_market_ou25",
        "goal_supremacy_form",
        "goal_supremacy_real",
        "goal_supremacy_error_market",
        "goal_supremacy_error_form",
        "rest_diff_days",
        "short_rest_home",
        "short_rest_away",
        "rest_advantage_home",
        "rest_advantage_away",
    ]
    cols_preview = [c for c in cols_preview if c in df.columns]
    print(df[cols_preview].head(10))

    # -------------------------------
    # Salvataggio
    # -------------------------------
    df.to_parquet(OUT_PATH, index=False)
    print(f"💾 Salvato: {OUT_PATH}")
    print(f"✅ STEP1C v4 COMPLETATO! (shape finale: {df.shape})")
    print("============================================")


if __name__ == "__main__":
    main()