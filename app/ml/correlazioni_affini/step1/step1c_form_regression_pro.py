# app/ml/correlazioni_affini/step1/step1c_form_regression_pro.py

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]      # .../app/ml/
AFFINI_DIR = BASE_DIR / "correlazioni_affini"
DATA_DIR = AFFINI_DIR / "data"

RAW_JSON = BASE_DIR.parents[1] / "data" / "all_matches_raw.json"  
OUTPUT_FILE = DATA_DIR / "step1c_form_regression.parquet"


# ---------------------------------------------------------
# LOAD RAW
# ---------------------------------------------------------
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
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------
# TREND REGRESSION
# ---------------------------------------------------------
def regression_trend(series):
    if len(series) < 3:
        return 0.0
    X = np.arange(len(series)).reshape(-1, 1)
    y = np.array(series)
    model = LinearRegression().fit(X, y)
    return float(model.coef_[0])


# ---------------------------------------------------------
# CORE FORM REGRESSION
# ---------------------------------------------------------
def compute_form_regression(df: pd.DataFrame):
    leagues = df["league"].unique()
    final_rows = []

    for L in tqdm(leagues, desc="League"):
        sub = df[df["league"] == L].copy()

        teams = sorted(
            pd.unique(sub[["home_team", "away_team"]].values.ravel())
        )

        history = {t: [] for t in teams}

        for _, r in sub.iterrows():
            ht = r.home_team
            at = r.away_team

            hist_ht = history[ht].copy()
            hist_at = history[at].copy()

            def stats(hist):
                if len(hist) == 0:
                    return dict(
                        pts=0, gf=0, ga=0, gd=0,
                        last5_pts=0, last5_gf=0, last5_ga=0, last5_gd=0,
                        trend_gf=0, trend_ga=0, trend_gd=0,
                        var_gf=0, var_ga=0, var_gd=0
                    )

                pts = sum(h["pts"] for h in hist)
                gf = sum(h["gf"] for h in hist)
                ga = sum(h["ga"] for h in hist)
                gd = gf - ga

                last5 = hist[-5:]

                last5_pts = sum(h["pts"] for h in last5)
                last5_gf = sum(h["gf"] for h in last5)
                last5_ga = sum(h["ga"] for h in last5)
                last5_gd = last5_gf - last5_ga

                trend_gf = regression_trend([h["gf"] for h in last5])
                trend_ga = regression_trend([h["ga"] for h in last5])
                trend_gd = regression_trend([h["gf"] - h["ga"] for h in last5])

                var_gf = np.var([h["gf"] for h in last5]) if len(last5) > 1 else 0
                var_ga = np.var([h["ga"] for h in last5]) if len(last5) > 1 else 0
                var_gd = np.var([(h["gf"] - h["ga"]) for h in last5]) if len(last5) > 1 else 0

                return dict(
                    pts=pts, gf=gf, ga=ga, gd=gd,
                    last5_pts=last5_pts, last5_gf=last5_gf,
                    last5_ga=last5_ga, last5_gd=last5_gd,
                    trend_gf=trend_gf, trend_ga=trend_ga, trend_gd=trend_gd,
                    var_gf=var_gf, var_ga=var_ga, var_gd=var_gd,
                )

            f_ht = stats(hist_ht)
            f_at = stats(hist_at)

            final_rows.append({
                "match_id": r.match_id,
                "date": r.date,
                "league": L,
                "home_team": ht,
                "away_team": at,
                "home_pts": f_ht["pts"],
                "home_gf": f_ht["gf"],
                "home_ga": f_ht["ga"],
                "home_gd": f_ht["gd"],
                "home_last5_pts": f_ht["last5_pts"],
                "home_last5_gf": f_ht["last5_gf"],
                "home_last5_ga": f_ht["last5_ga"],
                "home_last5_gd": f_ht["last5_gd"],
                "home_trend_gf": f_ht["trend_gf"],
                "home_trend_ga": f_ht["trend_ga"],
                "home_trend_gd": f_ht["trend_gd"],
                "home_var_gf": f_ht["var_gf"],
                "home_var_ga": f_ht["var_ga"],
                "home_var_gd": f_ht["var_gd"],
                "away_pts": f_at["pts"],
                "away_gf": f_at["gf"],
                "away_ga": f_at["ga"],
                "away_gd": f_at["gd"],
                "away_last5_pts": f_at["last5_pts"],
                "away_last5_gf": f_at["last5_gf"],
                "away_last5_ga": f_at["last5_ga"],
                "away_last5_gd": f_at["last5_gd"],
                "away_trend_gf": f_at["trend_gf"],
                "away_trend_ga": f_at["trend_ga"],
                "away_trend_gd": f_at["trend_gd"],
                "away_var_gf": f_at["var_gf"],
                "away_var_ga": f_at["var_ga"],
                "away_var_gd": f_at["var_gd"],
            })

            pts_home = 3 if r.home_ft > r.away_ft else 1 if r.home_ft == r.away_ft else 0
            pts_away = 3 if r.away_ft > r.home_ft else 1 if r.home_ft == r.away_ft else 0

            history[ht].append({"gf": r.home_ft, "ga": r.away_ft, "pts": pts_home})
            history[at].append({"gf": r.away_ft, "ga": r.home_ft, "pts": pts_away})

    return pd.DataFrame(final_rows)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("📥 STEP1C: Carico raw...")
    df = load_raw(RAW_JSON)

    print("⚙️ Calcolo Form Regression PRO...")
    out = compute_form_regression(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"💾 Salvo → {OUTPUT_FILE}")
    out.to_parquet(OUTPUT_FILE, index=False)

    print("✅ STEP1C Form Regression COMPLETATA!")


if __name__ == "__main__":
    main()