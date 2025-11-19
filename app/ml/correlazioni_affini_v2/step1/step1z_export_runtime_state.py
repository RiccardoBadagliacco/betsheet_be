# app/ml/correlazioni_affini_v2/check/check_step1z_runtime_state.py

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================
# PATH
# ============================================

BASE_DIR = Path(__file__).resolve().parents[2]  # .../app/ml
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"

IN_PATH = DATA_DIR / "affini_runtime_state.parquet"


def main():
    print("====================================================")
    print("📊 CHECK STEP1Z — RUNTIME STATE (Elo + Form smussata)")
    print("====================================================")
    print(f"📥 Input: {IN_PATH}")

    df = pd.read_parquet(IN_PATH)
    print(f"\n📏 Shape runtime_state: {df.shape[0]} righe, {df.shape[1]} colonne\n")

    # -----------------------------
    # 1) Colonne e tipi
    # -----------------------------
    print("🔢 ELENCO COLONNE:")
    for c in df.columns:
        print(f"  • {c}")
    print()

    print("🔤 TIPI DATI:")
    print(df.dtypes)
    print()

    # Assicuriamoci che le date siano datetime
    if "last_match_date" in df.columns:
        df["last_match_date"] = pd.to_datetime(df["last_match_date"], errors="coerce")

    # -----------------------------
    # 2) Unicità per team
    # -----------------------------
    if "team" in df.columns:
        n_teams = df["team"].nunique()
        print(f"🏟️  Squadre uniche: {n_teams}")
        dup_counts = df["team"].value_counts()
        max_rows_per_team = dup_counts.max()
        print(f"🔍 Righe per team (min/max): {dup_counts.min()} / {dup_counts.max()}")
        if max_rows_per_team > 1:
            print("🚨 ATTENZIONE: esistono team con più di una riga!")
            print(dup_counts[dup_counts > 1].head())
        else:
            print("✅ Ogni team ha esattamente una riga nel runtime state.")
    else:
        print("⚠️ Nessuna colonna 'team' trovata!")
    print()

    # -----------------------------
    # 3) NaN / valori mancanti
    # -----------------------------
    print("🧪 PERCENTUALI DI NULL PER COLONNA:")
    null_pct = df.isna().mean().sort_values(ascending=False) * 100
    print(null_pct)
    print()

    # -----------------------------
    # 4) STATISTICHE Elo
    # -----------------------------
    if "elo" in df.columns:
        print("📈 STATISTICHE ELO:")
        print(df["elo"].describe())
        print()

        n_elo_null = df["elo"].isna().sum()
        print(f"🔍 Elo null: {n_elo_null}")
        if n_elo_null > 0:
            print("   → ATTENZIONE: alcuni team non hanno Elo definito.")
    else:
        print("⚠️ Nessuna colonna 'elo' trovata!")
    print()

    # -----------------------------
    # 5) STATISTICHE FORM (aggregata)
    # -----------------------------
    form_cols = ["form_pts", "form_gf", "form_ga", "form_winrate", "form_matches"]
    present_form_cols = [c for c in form_cols if c in df.columns]

    if present_form_cols:
        print("📈 STATISTICHE FORM (aggregata):")
        print(df[present_form_cols].describe())
        print()

        # distribuzione form_matches
        if "form_matches" in df.columns:
            print("📊 DISTRIBUZIONE form_matches:")
            print(df["form_matches"].value_counts().sort_index().head(20))
            print()

            few_matches = df[df["form_matches"] < 3].shape[0]
            print(f"🔍 Team con form_matches < 3 (dopo smoothing): {few_matches}")
    else:
        print("⚠️ Colonne aggregare di form non trovate (form_pts / form_gf / form_ga / form_winrate / form_matches).")
    print()

    # -----------------------------
    # 6) STATISTICHE FORM HOME/AWAY (se presenti)
    # -----------------------------
    home_form_cols = [
        "form_pts_home", "form_gf_home", "form_ga_home",
        "form_winrate_home", "form_matches_home"
    ]
    away_form_cols = [
        "form_pts_away", "form_gf_away", "form_ga_away",
        "form_winrate_away", "form_matches_away"
    ]

    present_home = [c for c in home_form_cols if c in df.columns]
    present_away = [c for c in away_form_cols if c in df.columns]

    if present_home:
        print("📈 STATISTICHE FORM HOME:")
        print(df[present_home].describe())
        print()
    if present_away:
        print("📈 STATISTICHE FORM AWAY:")
        print(df[present_away].describe())
        print()

    # -----------------------------
    # 7) Date: min/max e distribuzione
    # -----------------------------
    if "last_match_date" in df.columns:
        print("📅 last_match_date (min/max):")
        print(df["last_match_date"].agg(["min", "max"]))
        print()

        # opzionale: distribuzione per anno
        print("📅 Distribuzione per anno last_match_date:")
        year_counts = df["last_match_date"].dt.year.value_counts().sort_index()
        print(year_counts)
        print()

    # -----------------------------
    # 8) Correlazioni rapide Elo/Form (diagnostica)
    # -----------------------------
    num_cols = []
    for c in ["elo", "form_pts", "form_gf", "form_ga", "form_winrate"]:
        if c in df.columns:
            num_cols.append(c)

    if len(num_cols) > 1:
        print("📌 MATRICE DI CORRELAZIONE (Elo vs Form aggregata):")
        corr = df[num_cols].corr()
        print(corr)
        print()

    print("🏁 CHECK STEP1Z COMPLETATO")
    print("====================================================")


if __name__ == "__main__":
    main()