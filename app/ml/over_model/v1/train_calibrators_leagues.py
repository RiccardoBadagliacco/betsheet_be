# app/ml/over_model/v1/train_calibrators_leagues.py
import os, math, sqlite3, pickle
import pandas as pd
from sklearn.isotonic import IsotonicRegression

DB_PATH = "./data/football_dataset.db"
OUT_DIR = os.path.join(os.path.dirname(__file__), "calibrators", "leagues")
os.makedirs(OUT_DIR, exist_ok=True)

def fair_prob_over25(O, U):
    inv = 1/O + 1/U
    return (1/O) / inv

def solve_lambda_from_pover25(p):
    lo, hi = 0.01, 6.0
    for _ in range(60):
        mid = (lo+hi)/2
        p0 = math.exp(-mid)
        p1 = p0*mid
        p2 = p1*(mid/2)
        if 1-(p0+p1+p2) < p: lo = mid
        else: hi = mid
    return (lo+hi)/2

def poisson_probs(l):
    p0 = math.exp(-l)
    p1 = p0*l
    p2 = p1*(l/2)
    return {"p05":1-p0, "p15":1-(p0+p1), "p25":1-(p0+p1+p2)}

conn = sqlite3.connect(DB_PATH)
q = """
SELECT l.code AS league_code,
       m.home_goals_ft AS FTHG, m.away_goals_ft AS FTAG,
       m.avg_over_25_odds AS O25, m.avg_under_25_odds AS U25
FROM matches m
JOIN seasons s ON m.season_id = s.id
JOIN leagues l ON s.league_id = l.id
WHERE m.avg_over_25_odds IS NOT NULL
  AND m.avg_under_25_odds IS NOT NULL
  AND m.home_goals_ft IS NOT NULL
  AND m.away_goals_ft IS NOT NULL
"""
df = pd.read_sql(q, conn)
conn.close()

df["goals"] = df.FTHG + df.FTAG
df = df[(df.O25>1.01)&(df.U25>1.01)].copy()

for lg, sub in df.groupby("league_code"):
    if len(sub) < 800:
        print(f"⚠️ skip {lg}: only {len(sub)} matches")
        continue
    rows=[]
    for _,r in sub.iterrows():
        p = fair_prob_over25(r.O25, r.U25)
        lam = solve_lambda_from_pover25(p)
        pr = poisson_probs(lam)
        rows.append({
            "p05":pr["p05"], "over05":int(r.goals>0),
            "p15":pr["p15"], "over15":int(r.goals>1),
            "p25":pr["p25"], "over25":int(r.goals>2),
        })
    df2 = pd.DataFrame(rows)
    print(f"League {lg}: {len(df2):,} matches")

    for col, tgt, suf in [("p05","over05","05"),("p15","over15","15"),("p25","over25","25")]:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(df2[col], df2[tgt])
        out = os.path.join(OUT_DIR, f"{lg}_{suf}.pkl")
        with open(out, "wb") as f: pickle.dump(iso, f)
        print(f"   ✅ saved {out}")

print("🎯 league calibrators done")
