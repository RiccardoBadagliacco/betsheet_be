# app/ml/correlazioni_affini_v2/check/check_step1c_dataset.py

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================
# PATH
# ============================================
BASE_DIR = Path(__file__).resolve().parents[2]  # .../app/ml
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"

IN_PATH = DATA_DIR / "step1c_dataset_with_elo_form.parquet"


def main():
    print("====================================================")
    print("📊 CHECK STEP1C — DATASET MASTER (base + Elo + Form)")
    print("====================================================")
    print(f"📥 Input: {IN_PATH}")

    df = pd.read_parquet(IN_PATH)
    print(f"\n📏 Shape dataset: {df.shape[0]} righe, {df.shape[1]} colonne\n")

    # -----------------------------------------
    # 1) Elenco colonne + raggruppamento logico
    # -----------------------------------------
    print("🔢 ELENCO COLONNE (tutte):")
    for c in df.columns:
        print(f"  • {c}")
    print()

    # Categorizziamo le colonne per "famiglia"
    col_base = [
        "match_id", "date", "league", "country", "season",
        "home_team", "away_team",
        "home_ft", "away_ft", "total_goals",
        "is_home_win", "is_draw", "is_away_win",
        "is_over05", "is_over15", "is_over25", "is_over35", "is_under25",
    ]
    col_base = [c for c in col_base if c in df.columns]

    col_market_1x2 = [c for c in df.columns if c.startswith("bk_") and "_ou" not in c]
    col_market_ou  = [c for c in df.columns if ("_ou25" in c or "bk_pO25" in c or "bk_pU25" in c or "entropy_bk_ou25" in c)]

    col_elo = [c for c in df.columns if c.startswith("elo_") or c in ["exp_home", "exp_away"]]
    col_form = [c for c in df.columns if "form_" in c and not c.startswith("season_")]
    col_team_strength = [c for c in df.columns if c.startswith("team_strength")]
    col_market_eng = [
        "entropy_bk_1x2",
        "fav_side_1x2",
        "fav_prob_1x2",
        "market_balance_index_1x2",
        "second_fav_prob_1x2",
        "fav_prob_gap_1x2",
        "spread_home_away_probs_1x2",
        "market_sign_vs_strength",
    ]
    col_market_eng = [c for c in col_market_eng if c in df.columns]

    col_ou25_eng = [
        "entropy_bk_ou25",
        "ou25_has_market",
        "ou25_bias_over",
        "ou25_balance_index",
    ]
    col_ou25_eng = [c for c in col_ou25_eng if c in df.columns]

    col_time = [
        "season_recency",
        "days_since_last_home",
        "days_since_last_away",
        "match_density_index",
    ]
    col_time = [c for c in col_time if c in df.columns]

    print("📦 Gruppi di colonne individuati:")
    print(f"  • Base match         : {len(col_base)}")
    print(f"  • Mercato 1X2 raw    : {len(col_market_1x2)}")
    print(f"  • Mercato OU2.5 raw  : {len(col_market_ou)}")
    print(f"  • Elo                : {len(col_elo)}")
    print(f"  • Form               : {len(col_form)}")
    print(f"  • Team strength      : {len(col_team_strength)}")
    print(f"  • Market features    : {len(col_market_eng)}")
    print(f"  • OU2.5 features     : {len(col_ou25_eng)}")
    print(f"  • Tempo / calendario : {len(col_time)}")
    print()

    # -----------------------------------------
    # 2) Tipi dati + null %
    # -----------------------------------------
    print("🔤 TIPI DATI:")
    print(df.dtypes)
    print()

    print("🧪 PERCENTUALI DI NULL PER COLONNA (descending):")
    null_pct = df.isna().mean().sort_values(ascending=False) * 100
    print(null_pct)
    print()

    # -----------------------------------------
    # 3) Statistiche di base sui target
    # -----------------------------------------
    print("📈 STATISTICHE RISULTATI REALI:")
    if {"home_ft", "away_ft", "total_goals"}.issubset(df.columns):
        print(df[["home_ft", "away_ft", "total_goals"]].describe())
    print()

    print("🏷️ FREQUENZE 1X2 REALI:")
    for c in ["is_home_win", "is_draw", "is_away_win"]:
        if c in df.columns:
            print(f"  {c}: {df[c].mean():.5f}")
    print()

    print("🏷️ FREQUENZE OVER/UNDER REALI:")
    for c in ["is_over05", "is_over15", "is_over25", "is_over35"]:
        if c in df.columns:
            print(f"  {c}: {df[c].mean():.5f}")
    if "is_under25" in df.columns:
        print(f"  is_under25: {df['is_under25'].mean():.5f}")
    print()

    # -----------------------------------------
    # 4) Statistiche rapide feature chiave
    # -----------------------------------------
    key_cols_num = [
        "bk_p1", "bk_px", "bk_p2",
        "bk_overround_1x2",
        "entropy_bk_1x2",
        "fav_prob_1x2",
        "elo_home_pre", "elo_away_pre", "elo_diff",
        "home_form_pts_avg_lastN", "away_form_pts_avg_lastN",
        "form_pts_diff", "form_gf_diff", "form_ga_diff", "form_win_rate_diff",
        "team_strength_home", "team_strength_away", "team_strength_diff",
        "market_balance_index_1x2", "fav_prob_gap_1x2", "spread_home_away_probs_1x2",
        "bk_pO25", "bk_pU25", "entropy_bk_ou25", "ou25_bias_over",
        "season_recency", "days_since_last_home", "days_since_last_away", "match_density_index",
    ]
    key_cols_num = [c for c in key_cols_num if c in df.columns]

    print("📈 STATISTICHE FEATURE CHIAVE:")
    if key_cols_num:
        print(df[key_cols_num].describe().T)
    print()

    # -----------------------------------------
    # 5) Correlazioni con i target (is_home_win, is_away_win, is_over25)
    # -----------------------------------------
    print("📌 CORRELAZIONI (assolute) con i target principali")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    targets = []
    for t in ["is_home_win", "is_away_win", "is_draw", "is_over25"]:
        if t in df.columns:
            targets.append(t)

    for target in targets:
        print("\n---------------------------------------------")
        print(f"🎯 Target: {target}")
        print("---------------------------------------------")

        corr = df[numeric_cols].corrwith(df[target])
        corr = corr.dropna().sort_values(key=lambda x: x.abs(), ascending=False)

        # escludiamo l'auto-correlazione ovvia se target è tra numeric_cols
        corr = corr.drop(labels=[target], errors="ignore")

        top_n = 20
        print(f"\n🔝 Top {top_n} feature per |corr| con {target}:")
        print(corr.head(top_n))

    print()

    # -----------------------------------------
    # 6) Distribuzione season_recency & timing
    # -----------------------------------------
    if "season_recency" in df.columns:
        print("📅 DISTRIBUZIONE season_recency:")
        print(df["season_recency"].describe())
        print("\nPer stagione:")
        if "season" in df.columns:
            print(df.groupby("season")["season_recency"].agg(["min", "max", "mean", "count"]))
        print()

    if {"days_since_last_home", "days_since_last_away"}.issubset(df.columns):
        print("⏱️ DISTRIBUZIONE days_since_last_home / away:")
        print(df[["days_since_last_home", "days_since_last_away"]].describe())
        print()

    if "match_density_index" in df.columns:
        print("📊 DISTRIBUZIONE match_density_index:")
        print(df["match_density_index"].describe())
        print()

    # -----------------------------------------
    # 7) Piccolo sanity check Elo vs esito 1X2
    # -----------------------------------------
    if {"elo_diff", "is_home_win", "is_away_win"}.issubset(df.columns):
        print("🧠 SANITY CHECK: Elo_diff vs outcome 1X2")
        bins = [-9999, -200, -100, -50, 0, 50, 100, 200, 9999]
        df["elo_diff_bin"] = pd.cut(df["elo_diff"], bins=bins)

        grp = df.groupby("elo_diff_bin").agg(
            n=("match_id", "count"),
            p_home=("is_home_win", "mean"),
            p_away=("is_away_win", "mean"),
            p_draw=("is_draw", "mean") if "is_draw" in df.columns else ("is_home_win", "mean"),
        )
        print(grp)
        df.drop(columns=["elo_diff_bin"], inplace=True)
        print()

    # -----------------------------------------
    # 8) Piccolo sanity check mercato 1X2 vs esito
    # -----------------------------------------
    if {"fav_side_1x2", "fav_prob_1x2", "is_home_win", "is_away_win"}.issubset(df.columns):
        print("💵 SANITY CHECK: favorito 1X2 vs esito")
        grp_fav = df.groupby("fav_side_1x2").agg(
            n=("match_id", "count"),
            fav_prob=("fav_prob_1x2", "mean"),
            p_home=("is_home_win", "mean"),
            p_away=("is_away_win", "mean"),
            p_draw=("is_draw", "mean") if "is_draw" in df.columns else ("is_home_win", "mean"),
        )
        print(grp_fav)
        print()

    print("🏁 CHECK STEP1C COMPLETATO")
    print("====================================================")


if __name__ == "__main__":
    main()