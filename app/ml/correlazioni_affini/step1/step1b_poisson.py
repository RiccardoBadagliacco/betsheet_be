# app/ml/correlazioni_affini/step1/step1b_poisson_expected_goals.py

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import poisson


# =========================================================
# PATH CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[2]  # .../app/ml/
AFFINI_DIR = BASE_DIR / "correlazioni_affini"
DATA_DIR = AFFINI_DIR / "data"

RAW_JSON = BASE_DIR.parents[1] / "data" / "all_matches_raw.json"

OUTPUT_FILE = DATA_DIR / "step1b_poisson_expected_goals.parquet"


# =========================================================
# LOAD RAW MATCHES
# =========================================================
def load_raw(path: Path) -> pd.DataFrame:
    print("📥 Carico dati raw...")
    with open(path, "r") as f:
        j = json.load(f)

    rows = []
    for mid, obj in j.items():
        m = obj.get("match", {})

        rows.append({
            "match_id": mid,
            "date": m.get("date"),
            "league": m.get("league", {}).get("code"),
            "season": m.get("season", {}).get("name"),
            "home_team": m.get("home_team", {}).get("name"),
            "away_team": m.get("away_team", {}).get("name"),
            "home_ft": m.get("result", {}).get("home_ft"),
            "away_ft": m.get("result", {}).get("away_ft"),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# =========================================================
# FIT POISSON STRENGTHS
# Identico al vecchio
# =========================================================
def fit_poisson_strengths(df: pd.DataFrame):
    strengths = {}
    alpha = 5.0  # smoothing identico

    for L, sub in df.groupby("league"):
        league_home_mean = sub["home_ft"].mean()
        league_away_mean = sub["away_ft"].mean()

        teams = sorted(pd.unique(sub[["home_team", "away_team"]].values.ravel()))
        team_stats = {}

        for t in teams:
            mask_h = sub["home_team"] == t
            mask_a = sub["away_team"] == t

            home_matches = mask_h.sum()
            away_matches = mask_a.sum()

            home_goals_for = sub.loc[mask_h, "home_ft"].sum()
            home_goals_against = sub.loc[mask_h, "away_ft"].sum()

            away_goals_for = sub.loc[mask_a, "away_ft"].sum()
            away_goals_against = sub.loc[mask_a, "home_ft"].sum()

            gsh = (home_goals_for + alpha * league_home_mean) / (home_matches + alpha)
            gch = (home_goals_against + alpha * league_away_mean) / (home_matches + alpha)

            gas = (away_goals_for + alpha * league_away_mean) / (away_matches + alpha)
            gac = (away_goals_against + alpha * league_home_mean) / (away_matches + alpha)

            team_stats[t] = dict(gsh=gsh, gch=gch, gas=gas, gac=gac)

        strengths[L] = {
            "team_stats": team_stats,
            "league_home_mean": league_home_mean,
            "league_away_mean": league_away_mean,
        }

    return strengths


# =========================================================
# EXPECTED GOALS CALCULATION
# Identico al vecchio
# =========================================================
def compute_expected(df: pd.DataFrame, strengths):
    rows = []

    for _, r in tqdm(df.iterrows(), total=len(df)):
        L = r.league
        info = strengths.get(L)
        if info is None:
            continue

        ts = info["team_stats"]
        home_stats = ts.get(r.home_team)
        away_stats = ts.get(r.away_team)

        if home_stats is None or away_stats is None:
            continue

        lam_home = (home_stats["gsh"] + away_stats["gac"]) / 2
        lam_away = (away_stats["gas"] + home_stats["gch"]) / 2

        lam_home = float(np.clip(lam_home, 0.2, 4.0))
        lam_away = float(np.clip(lam_away, 0.2, 4.0))
        lam_tot = lam_home + lam_away

        prob_under25 = sum(poisson.pmf(g, lam_tot) for g in [0, 1, 2])
        prob_over25 = 1 - prob_under25

        prob_gg = 1 - (poisson.pmf(0, lam_home) * poisson.pmf(0, lam_away))
        prob_ng = 1 - prob_gg

        home_probs = {f"p_home_{g}": poisson.pmf(g, lam_home) for g in range(6)}
        away_probs = {f"p_away_{g}": poisson.pmf(g, lam_away) for g in range(6)}

        rows.append({
            "match_id": r.match_id,
            "exp_goals_home": lam_home,
            "exp_goals_away": lam_away,
            "prob_over25": prob_over25,
            "prob_under25": prob_under25,
            "prob_gg": prob_gg,
            "prob_ng": prob_ng,
            **home_probs,
            **away_probs,
        })

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================
def main():
    print("📥 STEP1B: Carico JSON raw...")
    df = load_raw(RAW_JSON)

    print("⚙️ Calcolo Poisson strengths...")
    strengths = fit_poisson_strengths(df)

    print("🔥 Calcolo expected goals...")
    df_exp = compute_expected(df, strengths)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"💾 Salvo → {OUTPUT_FILE}")
    df_exp.to_parquet(OUTPUT_FILE, index=False)

    print("✅ STEP1B Poisson completato!")


if __name__ == "__main__":
    main()