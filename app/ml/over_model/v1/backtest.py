
# final_backtest.py (versione ridotta per OverModelV1)
import os, sqlite3, pandas as pd, numpy as np
from tqdm import tqdm
from model import OverModelV1

DB_PATH = "./data/football_dataset.db"

def brier(y, p): 
    y = np.array(y, float); p = np.array(p, float)
    return float(np.mean((p-y)**2))

def load_matches(limit=None):
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT l.code AS league_code,
           m.match_date AS Date,
           ht.name AS HomeTeam, at.name AS AwayTeam,
           m.home_goals_ft AS FTHG, m.away_goals_ft AS FTAG,
           m.avg_over_25_odds AS "Avg>2.5",
           m.avg_under_25_odds AS "Avg<2.5"
    FROM matches m
    JOIN teams ht ON m.home_team_id=ht.id
    JOIN teams at ON m.away_team_id=at.id
    JOIN seasons s ON m.season_id=s.id
    JOIN leagues l ON s.league_id=l.id
    WHERE m.avg_over_25_odds IS NOT NULL
      AND m.avg_under_25_odds IS NOT NULL
      AND m.home_goals_ft IS NOT NULL
      AND m.away_goals_ft IS NOT NULL
    ORDER BY m.match_date ASC
    """
    df = pd.read_sql(q, conn)
    conn.close()
    if limit: df = df.tail(limit)
    return df

def _brier_thr(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2)) if len(y) else float('nan')

def _sweep_one_thr(y, p, thresholds=None, min_coverage=0.0):
    if thresholds is None:
        thresholds = np.round(np.linspace(0.50, 0.90, 41), 2)

    rows = []
    brier_all = _brier_thr(y, p)

    for t in thresholds:
        sel = p >= t
        cov = float(sel.mean())

        if cov < min_coverage or sel.sum() == 0:
            rows.append({
                "threshold": t, "coverage": cov,
                "n_selected": int(sel.sum()),
                "accuracy": np.nan,
                "brier_selected": np.nan,
                "brier_all": brier_all
            })
            continue

        y_sel = y[sel]; p_sel = p[sel]
        acc = float(((p_sel >= t).astype(int) == y_sel.astype(int)).mean())
        brier_sel = _brier_thr(y_sel, p_sel)

        rows.append({
            "threshold": t,
            "coverage": cov,
            "n_selected": int(sel.sum()),
            "accuracy": acc,
            "brier_selected": brier_sel,
            "brier_all": brier_all
        })

    return pd.DataFrame(rows)

def best_threshold(df_sweep, objective="max_accuracy", min_coverage=0.0):
    df = df_sweep.copy()
    df = df[np.isfinite(df["accuracy"])]
    if min_coverage > 0:
        df = df[df["coverage"] >= min_coverage]
    if df.empty:
        return None

    if objective == "max_accuracy":
        df = df.sort_values(["accuracy","coverage"], ascending=[False,False])
    else:
        df = df.sort_values(["brier_selected","coverage"], ascending=[True,False])

    return df.iloc[0].to_dict()

def sweep_all_markets(out_df, min_coverage=0.05, objective="max_accuracy"):
    results = {}
    for label, y, p in [
        ("O0.5","y05","p05"),
        ("O1.5","y15","p15"),
        ("O2.5","y25","p25"),
    ]:
        sw = _sweep_one_thr(out_df[y], out_df[p], min_coverage=min_coverage)
        best = best_threshold(sw, min_coverage=min_coverage, objective=objective)
        results[label] = {"best": best, "table": sw}
    return results

def print_thr_results(res):
    print("\n🎯 Best thresholds:")
    for m in ["O0.5","O1.5","O2.5"]:
        b = res[m]["best"]
        if not b:
            print(f"{m}: no result")
            continue
        print(f"{m}: thr={b['threshold']:.2f} | cov={b['coverage']*100:.1f}% "
              f"| acc={b['accuracy']:.3f} | Brier(sel)={b['brier_selected']:.4f}")

if __name__ == "__main__":
    df = load_matches(limit=30000)
    model = OverModelV1(debug=False)

    rows=[]
    for _, r in tqdm(df.iterrows(), total=len(df)):
        m = {
            "Avg>2.5": float(r["Avg>2.5"]),
            "Avg<2.5": float(r["Avg<2.5"]),
            "league_code": r["league_code"]
        }
        pred = model.predict(m)
        if not pred: continue
        tot = int(r["FTHG"])+int(r["FTAG"])
        rows.append({
            "p05": pred["p_over_0_5"], "y05": int(tot>0),
            "p15": pred["p_over_1_5"], "y15": int(tot>1),
            "p25": pred["p_over_2_5"], "y25": int(tot>2),
        })

    out = pd.DataFrame(rows)
    for name, y, p in [("Over 0.5","y05","p05"),("Over 1.5","y15","p15"),("Over 2.5","y25","p25")]:
        print(f"{name:<10} Brier Score: {brier(out[y], out[p]):.4f}")
        acc = float(( (out[p]>=0.5) == (out[y]==1) ).mean())
        print(f"{name:<10} Accuracy: {acc:.4f}")
    
    # ✅ now run threshold sweep
    """ for c in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
        print(f"\n--- Threshold sweep (min coverage: {c*100:.1f}%) ---")
        thr = sweep_all_markets(out, min_coverage=c, objective="max_accuracy")
        print_thr_results(thr) """
        
    # ✅ Collect best thresholds (min coverage 10% per esempio)
    best = sweep_all_markets(out, min_coverage=0.10, objective="max_accuracy")

    thr_o05 = best["O0.5"]["best"]["threshold"]
    thr_o15 = best["O1.5"]["best"]["threshold"]
    thr_o25 = best["O2.5"]["best"]["threshold"]

    print("\n✅ FINAL SELECTED THRESHOLDS")
    print(f"O0.5: {thr_o05}, O1.5: {thr_o15}, O2.5: {thr_o25}")

    # Save to JSON for API to load
    import json, os
    os.makedirs("data", exist_ok=True)
    with open("data/over_thresholds.json", "w") as f:
        json.dump({
            "over_0_5": thr_o05,
            "over_1_5": thr_o15,
            "over_2_5": thr_o25
        }, f, indent=2)

    print("💾 Saved threshold config to data/over_thresholds.json")
    
    print("\n\n=== Evaluate ONLY recommended bets ===")

    # Load thresholds from best sweep results already computed
    thr05 = 0.90
    thr15 = 0.80
    thr25 = 0.68

    print(f"Using thresholds: O0.5={thr05}, O1.5={thr15}, O2.5={thr25}")

    # Filter matches where model would recommend betting
    sel05 = out[out["p05"] >= thr05]
    sel15 = out[out["p15"] >= thr15]
    sel25 = out[out["p25"] >= thr25]

    def eval_subset(df, label, pcol, ycol):
        if len(df) == 0:
            print(f"{label}: no selections")
            return
        b = brier(df[ycol], df[pcol])
        acc = float(((df[pcol] >= 0.5) == (df[ycol] == 1)).mean())
        print(f"{label:<12} | n={len(df)} | Brier={b:.4f} | Acc={acc:.4f}")

    eval_subset(sel05, "Over 0.5", "p05", "y05")
    eval_subset(sel15, "Over 1.5", "p15", "y15")
    eval_subset(sel25, "Over 2.5", "p25", "y25")


