# ============================================================
# app/ml/profeta/step0_profeta_sanity_check.py
# ============================================================

"""
Sanity check per profeta_step0.parquet

Controlla:
  - presenza colonne chiave
  - valori mancanti critici
  - età e pesi coerenti
  - coerenza goals vs is_fixture
  - distribuzione di leagues / seasons / team-season
  - statistiche base sui gol
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# PATH
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STEP0_PATH = DATA_DIR / "step0_profeta.parquet"


# ------------------------------------------------------------
# SANITY CHECK
# ------------------------------------------------------------

def sanity_check_step0(path: Path = STEP0_PATH):
    print("🔍 Sanity check PROFETA STEP0")
    print(f"📦 File: {path}")

    if not path.exists():
        print("❌ ERRORE: file profeta_step0.parquet non trovato.")
        return

    df = pd.read_parquet(path)
    print(f"✅ Caricato df: {df.shape[0]} righe, {df.shape[1]} colonne\n")

    # --------------------------------------------------------
    # 1) Colonne attese
    # --------------------------------------------------------
    expected_cols = [
        "match_id",
        "is_fixture",
        "match_date",
        "match_year",
        "age_years",
        "weight",
        "league_id",
        "league_code",
        "season_id",
        "season_code",
        "home_team_id",
        "away_team_id",
        "home_team_season_id",
        "away_team_season_id",
        "home_goals",
        "away_goals",
    ]

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        print("❌ ERRORE: mancano colonne attese:", missing_cols)
        return
    else:
        print("✅ Tutte le colonne attese sono presenti.")

    # --------------------------------------------------------
    # 2) Null e campi critici
    # --------------------------------------------------------
    print("\n📊 Null count (prime colonne chiave):")
    print(df[[
        "match_id",
        "is_fixture",
        "match_date",
        "league_id",
        "season_id",
        "home_team_id",
        "away_team_id",
        "home_team_season_id",
        "away_team_season_id",
    ]].isna().sum())

    # Match storici (is_fixture == False) devono avere tutto pieno
    hist = df[df["is_fixture"] == False]
    fix = df[df["is_fixture"] == True]

    print(f"\n🔢 Match storici: {len(hist)}")
    print(f"🔢 Fixture:       {len(fix)}\n")

    # Campi critici per match storici
    crit_hist_cols = [
        "match_date",
        "league_id",
        "season_id",
        "home_team_id",
        "away_team_id",
        "home_team_season_id",
        "away_team_season_id",
        "home_goals",
        "away_goals",
    ]

    bad_hist = hist[crit_hist_cols].isna().any(axis=1)
    if bad_hist.any():
        print(f"⚠️ ATTENZIONE: {bad_hist.sum()} match storici hanno NULL in campi critici.")
        print("   Esempio righe problematiche:")
        print(hist[bad_hist].head(10))
    else:
        print("✅ Match storici: nessun NULL in campi critici.")

    # Fixture: possiamo tollerare qualche NULL (season/league se non mappate),
    # ma è utile saperlo
    print("\n📊 Null nelle fixture (info diagnostica):")
    print(fix[[
        "match_date",
        "league_id",
        "season_id",
        "home_team_id",
        "away_team_id",
        "home_team_season_id",
        "away_team_season_id",
    ]].isna().sum())

    # --------------------------------------------------------
    # 3) age_years e weight
    # --------------------------------------------------------
    print("\n⏳ Controllo age_years e weight…")

    if (df["age_years"] < 0).any():
        print("❌ ERRORE: trovati age_years negativi!")
        print(df[df["age_years"] < 0][["match_id", "match_date", "age_years"]].head())
    else:
        print("✅ Nessun age_years negativo.")

    print("📈 Stats age_years (storici):")
    print(hist["age_years"].describe())

    print("\n⚖️ Stats weight (tutto il dataset):")
    print(df["weight"].describe())

    if (df["weight"] <= 0).any():
        print("❌ ERRORE: trovati weight <= 0!")
        print(df[df["weight"] <= 0][["match_id", "age_years", "weight"]].head())

    if (df["weight"] > 1.000001).any():
        print("❌ ERRORE: trovati weight > 1.0!")
        print(df[df["weight"] > 1.000001][["match_id", "age_years", "weight"]].head())

    # Fixture: age_years dovrebbe essere 0, weight 1
    bad_fix_age = fix[(fix["age_years"] != 0)]
    bad_fix_w = fix[(fix["weight"] != 1.0)]

    if len(bad_fix_age) > 0:
        print(f"⚠️ ATTENZIONE: {len(bad_fix_age)} fixture con age_years != 0")
        print(bad_fix_age[["match_id", "match_date", "age_years"]].head())
    else:
        print("✅ Fixture: age_years tutti 0.")

    if len(bad_fix_w) > 0:
        print(f"⚠️ ATTENZIONE: {len(bad_fix_w)} fixture con weight != 1.0")
        print(bad_fix_w[["match_id", "match_date", "age_years", "weight"]].head())
    else:
        print("✅ Fixture: weight tutti = 1.0.")

    # --------------------------------------------------------
    # 4) Goals: distribuzione e coerenza
    # --------------------------------------------------------
    print("\n⚽ Distribuzione gol (storici):")

    valid_hist = hist.dropna(subset=["home_goals", "away_goals"])
    print("📈 home_goals describe:")
    print(valid_hist["home_goals"].describe())
    print("\n📈 away_goals describe:")
    print(valid_hist["away_goals"].describe())

    if (valid_hist["home_goals"] < 0).any() or (valid_hist["away_goals"] < 0).any():
        print("❌ ERRORE: trovati gol negativi!")
        print(valid_hist[(valid_hist["home_goals"] < 0) | (valid_hist["away_goals"] < 0)].head())

    # --------------------------------------------------------
    # 5) Unicità e cardinalità di league / season / team-season
    # --------------------------------------------------------
    print("\n🏟  Cardinalità entità:")

    print(f"  • Leagues:       {df['league_id'].nunique()} (codici: {df['league_code'].nunique()})")
    print(f"  • Seasons:       {df['season_id'].nunique()}")
    print(f"  • Home teams:    {df['home_team_id'].nunique()}")
    print(f"  • Away teams:    {df['away_team_id'].nunique()}")
    print(f"  • Team-season (home): {df['home_team_season_id'].nunique()}")
    print(f"  • Team-season (away): {df['away_team_season_id'].nunique()}")

    # --------------------------------------------------------
    # 6) Check formato team-season-id
    # --------------------------------------------------------
    print("\n🧩 Check formato team-season id…")

    def check_ts_format(s: pd.Series):
        return s.dropna().apply(lambda x: "::" in str(x)).all()

    ok_home_ts = check_ts_format(df["home_team_season_id"])
    ok_away_ts = check_ts_format(df["away_team_season_id"])

    if not ok_home_ts or not ok_away_ts:
        print("⚠️ ATTENZIONE: qualche team-season id non contiene '::'")
        print(df[~df["home_team_season_id"].astype(str).str.contains("::")][
            ["match_id", "home_team_season_id"]
        ].head())
    else:
        print("✅ Formato team-season id coerente ('team_id::season_id').")

    # --------------------------------------------------------
    # 7) Riassunto finale
    # --------------------------------------------------------
    print("\n✅ Sanity check STEP0 completato.")
    print("   Se non hai visto '❌ ERRORE', Profeta può passare allo STEP1 😉")


def main():
    sanity_check_step0()


if __name__ == "__main__":
    main()