# ============================================================
# app/ml/profeta/step4_profeta_eval.py
# ============================================================

"""
STEP4 — Valutazione Profeta V0 su partite storiche

Input:
    data/step0_profeta.parquet
    data/step3_profeta_predictions.parquet

Output (print a console):
    - Brier e log-loss 1X2
    - Brier OU2.5
    - calibrazione per bucket
    - qualche statistica per lega
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR / "data"
STEP0_PATH = DATA_DIR / "step0_profeta.parquet"
STEP3_PATH = DATA_DIR / "step3_profeta_predictions.parquet"


def brier_multiclass(prob_cols, y_true, classes):
    """
    Brier score multiclass (1X2).
    prob_cols: df[["p1","px","p2"]]
    y_true:    serie con valori in classes (es ["1","X","2"])
    classes:   lista nell'ordine delle prob (["1","X","2"])
    """
    Y = np.zeros((len(y_true), len(classes)), dtype=float)
    for i, c in enumerate(classes):
        Y[:, i] = (y_true == c).astype(float)

    P = prob_cols.to_numpy()
    return np.mean(np.sum((P - Y) ** 2, axis=1))


def logloss_multiclass(prob_cols, y_true, classes, eps=1e-15):
    """
    Log-loss multiclass.
    """
    P = prob_cols.to_numpy().clip(eps, 1 - eps)
    cls_idx = {c: i for i, c in enumerate(classes)}
    idx = np.array([cls_idx[v] for v in y_true])
    p_true = P[np.arange(len(y_true)), idx]
    return -np.mean(np.log(p_true))


def brier_binary(p, y):
    """
    Brier binary (y ∈ {0,1})
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.mean((p - y) ** 2)


def main():
    print("📥 Carico STEP0:", STEP0_PATH)
    df0 = pd.read_parquet(STEP0_PATH)

    print("📥 Carico STEP3:", STEP3_PATH)
    df3 = pd.read_parquet(STEP3_PATH)

    print("🔗 Merge su match_id…")
    df = df0.merge(df3, on="match_id", suffixes=("_step0", "_step3"))

    # Filtra solo storici con gol not null
    df = df[(df["is_fixture_step0"] == False)]
    df = df.dropna(subset=["home_goals", "away_goals"])

    print("🔢 Partite storiche per valutazione:", len(df))

    # --------------------------------------------------------
    # Etichette reali 1X2
    # --------------------------------------------------------
    def outcome_1x2(row):
        if row["home_goals"] > row["away_goals"]:
            return "1"
        elif row["home_goals"] < row["away_goals"]:
            return "2"
        else:
            return "X"

    df["y_1x2"] = df.apply(outcome_1x2, axis=1)

    # Etichette reali OU2.5
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["y_ou25"] = (df["total_goals"] >= 3).astype(int)  # 1=Over, 0=Under

    # --------------------------------------------------------
    # Metriche 1X2
    # --------------------------------------------------------
    prob_1x2_cols = df[["p1", "px", "p2"]]
    y_1x2 = df["y_1x2"].values
    classes_1x2 = ["1", "X", "2"]

    brier_1x2 = brier_multiclass(prob_1x2_cols, y_1x2, classes_1x2)
    ll_1x2    = logloss_multiclass(prob_1x2_cols, y_1x2, classes_1x2)

    print("\n📊 METRICHE 1X2")
    print(f"  Brier 1X2:    {brier_1x2:.5f}")
    print(f"  Log-loss 1X2: {ll_1x2:.5f}")

    # Frequenze reali vs medie previste
    freq_real_1 = (df["y_1x2"] == "1").mean()
    freq_real_x = (df["y_1x2"] == "X").mean()
    freq_real_2 = (df["y_1x2"] == "2").mean()

    print("\n  Frequenze reali 1X2:")
    print(f"    1: {freq_real_1:.3%}")
    print(f"    X: {freq_real_x:.3%}")
    print(f"    2: {freq_real_2:.3%}")

    print("\n  Medie previste 1X2:")
    print(f"    1: {df['p1'].mean():.3%}")
    print(f"    X: {df['px'].mean():.3%}")
    print(f"    2: {df['p2'].mean():.3%}")

    # --------------------------------------------------------
    # Metriche OU2.5
    # --------------------------------------------------------
    p_over25 = df["p_over_2_5"].values
    y_ou25   = df["y_ou25"].values

    brier_ou25 = brier_binary(p_over25, y_ou25)

    print("\n📊 METRICHE OU2.5")
    print(f"  Brier OU2.5: {brier_ou25:.5f}")

    print("\n  Frequenza reale Over 2.5:", y_ou25.mean(), "→", f"{y_ou25.mean():.3%}")
    print("  Media prevista p_over_2_5:", df["p_over_2_5"].mean(), "→", f"{df['p_over_2_5'].mean():.3%}")

    # --------------------------------------------------------
    # CALIBRAZIONE (semplice) 1X2 sul segno 1
    # --------------------------------------------------------
    print("\n📈 Calibrazione grossolana su segno 1:")

    bins = np.linspace(0.0, 1.0, 11)
    df["p1_bin"] = pd.cut(df["p1"], bins=bins, include_lowest=True)

    calib = (
        df.groupby("p1_bin")
        .agg(
            mean_p1=("p1", "mean"),
            freq_win=("y_1x2", lambda x: (x == "1").mean()),
            count=("match_id", "size"),
        )
        .reset_index()
    )

    print(calib.to_string(index=False))

    # --------------------------------------------------------
    # SANITY PER LEAGUE: gol previsti vs reali
    # --------------------------------------------------------
    print("\n🏟  Gol medi per lega (reali vs previsti):")

    df["xg_total"] = df["lambda_home"] + df["lambda_away"]

    league_stats = (
        df.groupby("league_id_step0")
        .agg(
            real_goals=("total_goals", "mean"),
            pred_goals=("xg_total", "mean"),
            n=("match_id", "size"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
        .head(10)
    )

    print(league_stats.to_string(index=False))

    print("\n✅ STEP4 PROFETA — Valutazione completata.")

    # ============================================================
    # SKILL SCORE – Valutazione della bontà del modello
    # ============================================================

    print("\n\n====================================================")
    print("📈 INDICE DI BONTA' DEL MODELLO — PROFETA V0")
    print("====================================================")

    # --------------------------
    # Frequenze reali
    # --------------------------
    freq_1 = (df["y_1x2"] == "1").mean()
    freq_X = (df["y_1x2"] == "X").mean()
    freq_2 = (df["y_1x2"] == "2").mean()

    print(f"Frequenze reali:")
    print(f"   1: {freq_1:.3%}")
    print(f"   X: {freq_X:.3%}")
    print(f"   2: {freq_2:.3%}")

    # --------------------------
    # Probabilità baseline
    # --------------------------
    df["p1_base"] = freq_1
    df["px_base"] = freq_X
    df["p2_base"] = freq_2

    # --------------------------
    # Brier modello
    # --------------------------
    brier_model = np.mean(
        (df["p1"] - (df["y_1x2"]=="1").astype(float))**2 +
        (df["px"] - (df["y_1x2"]=="X").astype(float))**2 +
        (df["p2"] - (df["y_1x2"]=="2").astype(float))**2
    )

    # --------------------------
    # Brier baseline
    # --------------------------
    brier_base = np.mean(
        (df["p1_base"] - (df["y_1x2"]=="1").astype(float))**2 +
        (df["px_base"] - (df["y_1x2"]=="X").astype(float))**2 +
        (df["p2_base"] - (df["y_1x2"]=="2").astype(float))**2
    )

    brier_skill = 1 - (brier_model / brier_base)

    print("\n📊 Brier Skill Score")
    print(f"   Brier modello:  {brier_model:.6f}")
    print(f"   Brier baseline: {brier_base:.6f}")
    print(f"   👉 BSS: {brier_skill:.2%}")

    # --------------------------
    # Log-loss baseline
    # --------------------------
    eps = 1e-12
    # log-loss modello già calcolato in ll_1x2
    LL_model = ll_1x2

    LL_base = -np.mean([
        np.log([freq_1, freq_X, freq_2][["1","X","2"].index(y)] + eps)
        for y in df["y_1x2"]
    ])

    logloss_skill = 1 - (LL_model / LL_base)

    print("\n📊 Log-loss Skill Score")
    print(f"   LL modello:  {LL_model:.6f}")
    print(f"   LL baseline: {LL_base:.6f}")
    print(f"   👉 LLSS: {logloss_skill:.2%}")

    print("\n====================================================")
    print("📌 Interpretazione:")
    print("   0–5%   → molto debole")
    print("   5–10%  → discreto")
    print("   10–20% → buono")
    print("   20–30% → molto buono")
    print("   >30%   → eccezionale (quasi impossibile senza forti feature)")
    print("====================================================\n")


if __name__ == "__main__":
    main()
