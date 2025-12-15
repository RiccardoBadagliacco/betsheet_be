# ============================================================
# app/ml/profeta_v3/step0b_profeta_form.py (VERSIONE FINALE CORRETTA)
# ============================================================

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR / "data"
STEP0_PATH = DATA_DIR / "step0_profeta.parquet"


# ------------------------------------------------------------
# CALCOLO FORMA PER UN TEAM
# ------------------------------------------------------------
def compute_team_form(df_team):
    df_team = df_team.sort_values("match_date").reset_index(drop=True)

    gf, ga, pts = [], [], []

    # compute g_for / g_against / pts
    for _, r in df_team.iterrows():
        gf.append(r["g_for"])
        ga.append(r["g_against"])

        if r["g_for"] > r["g_against"]:
            pts.append(3)
        elif r["g_for"] == r["g_against"]:
            pts.append(1)
        else:
            pts.append(0)

    gf = pd.Series(gf)
    ga = pd.Series(ga)
    pts = pd.Series(pts)

    # LAST 5
    df_team["gf_last5"] = gf.shift(1).rolling(5).sum()
    df_team["ga_last5"] = ga.shift(1).rolling(5).sum()
    df_team["pts_last5"] = pts.shift(1).rolling(5).sum()
    df_team["gd_last5"] = df_team["gf_last5"] - df_team["ga_last5"]

    # LAST 10
    df_team["gf_last10"] = gf.shift(1).rolling(10).sum()
    df_team["ga_last10"] = ga.shift(1).rolling(10).sum()
    df_team["pts_last10"] = pts.shift(1).rolling(10).sum()
    df_team["gd_last10"] = df_team["gf_last10"] - df_team["ga_last10"]

    # Averages (useful later)
    df_team["avg_gf_last5"] = df_team["gf_last5"] / 5
    df_team["avg_ga_last5"] = df_team["ga_last5"] / 5

    return df_team


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    print("📥 Carico STEP0:", STEP0_PATH)
    df = pd.read_parquet(STEP0_PATH)

    hist = df[df["is_fixture"] == False].copy()
    print(f"🔧 Calcolo forma su {len(hist)} match storici…")

    # --------------------------------------------------------
    # COSTRUISCO DATAFRAME TEAM-MATCH (2 righe per match)
    # --------------------------------------------------------
    rows = []
    for _, r in hist.iterrows():
        # home team
        rows.append({
            "match_id": r["match_id"],
            "team_id": r["home_team_id"],
            "match_date": r["match_date"],
            "side": "home",
            "g_for": r["home_goals"],
            "g_against": r["away_goals"],
        })

        # away team
        rows.append({
            "match_id": r["match_id"],
            "team_id": r["away_team_id"],
            "match_date": r["match_date"],
            "side": "away",
            "g_for": r["away_goals"],
            "g_against": r["home_goals"],
        })

    df_team = pd.DataFrame(rows)

    # --------------------------------------------------------
    # CALCOLO FORMA PER TEAM
    # --------------------------------------------------------
    out = (
        df_team.groupby("team_id", group_keys=False)
        .apply(compute_team_form)
        .reset_index(drop=True)
    )

    # colonne finali di forma
    form_cols = [
        "gf_last5", "ga_last5", "pts_last5", "gd_last5",
        "gf_last10", "ga_last10", "pts_last10", "gd_last10",
        "avg_gf_last5", "avg_ga_last5",
    ]

    # --------------------------------------------------------
    # SEPARO HOME E AWAY
    # --------------------------------------------------------
    df_home = out[out["side"] == "home"][["match_id"] + form_cols]
    df_home = df_home.rename(columns={c: c + "_home" for c in form_cols})

    df_away = out[out["side"] == "away"][["match_id"] + form_cols]
    df_away = df_away.rename(columns={c: c + "_away" for c in form_cols})

    # --------------------------------------------------------
    # MERGE DEFINITIVO (UNA SOLA RIGA PER MATCH)
    # --------------------------------------------------------
    df_final = (
        df.merge(df_home, on="match_id", how="left")
          .merge(df_away, on="match_id", how="left")
    )

    print("💾 Scrivo STEP0 aggiornato…")
    df_final.to_parquet(STEP0_PATH, index=False)
    print("✅ STEP0B COMPLETATO — Forma home/away corretta!")


if __name__ == "__main__":
    main()