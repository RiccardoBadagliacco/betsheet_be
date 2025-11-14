# app/ml/correlazioni_affini/step4_profile_gmm_clusters.py

import pandas as pd
from pathlib import Path

# =============================================================================
# PATHS – FIXATI DEFINITIVAMENTE
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1] / "correlazioni_affini" / "data"

CLUSTERS = BASE_DIR / "step3_cluster_assignments.parquet"
RESULTS = BASE_DIR / "step0_dataset_base.parquet"

# =============================================================================
# UTILS
# =============================================================================

def pct(series_bool):
    """Percentuale [0–100] di True in una serie booleana."""
    if len(series_bool) == 0:
        return 0.0
    return 100.0 * series_bool.mean()

# =============================================================================
# MAIN
# =============================================================================

def main():

    print(f"DEBUG — CLUSTERS PATH: {CLUSTERS}")
    print(f"DEBUG — RESULTS PATH:  {RESULTS}")
    print(f"DEBUG — Exists clusters? {CLUSTERS.exists()}")
    print(f"DEBUG — Exists results?  {RESULTS.exists()}")

    if not CLUSTERS.exists():
        raise FileNotFoundError(f"File cluster non trovato: {CLUSTERS}")

    if not RESULTS.exists():
        raise FileNotFoundError(f"File risultati non trovato: {RESULTS}")

    # -------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------

    print("📥 Carico cluster_assignments...")
    cl = pd.read_parquet(CLUSTERS)

    print("📥 Carico dataset base con risultati reali...")
    base = pd.read_parquet(RESULTS)

    # -------------------------------------------------------------------------
    # Merge (uguale alla versione originale)
    # -------------------------------------------------------------------------

    print("🔗 Merge cluster_assignments + risultati...")

    df = cl.merge(
        base[["match_id", "home_ft", "away_ft"]],
        on="match_id",
        how="left",
        suffixes=("", "_real")
    )

    # Usa sempre i risultati reali dello STEP0
    df["home_ft"] = df["home_ft_real"]
    df["away_ft"] = df["away_ft_real"]
    df = df.drop(columns=["home_ft_real", "away_ft_real"])

    # Check
    if df["home_ft"].isna().any():
        print("⚠️ Ci sono match senza risultati reali. Verranno esclusi.")
        df = df.dropna(subset=["home_ft", "away_ft"])

    # Conversioni
    df["home_ft"] = df["home_ft"].astype(int)
    df["away_ft"] = df["away_ft"].astype(int)

    # Colonne derivate
    df["total_goals"] = df["home_ft"] + df["away_ft"]
    df["result_1x2"] = df.apply(
        lambda r: "1" if r["home_ft"] > r["away_ft"]
        else "2" if r["home_ft"] < r["away_ft"]
        else "X",
        axis=1
    )

    df["goal_diff"] = df["home_ft"] - df["away_ft"]
    df["abs_diff"] = df["goal_diff"].abs()

    clusters = sorted(df["cluster"].unique())

    print("\n📊 Profilazione CLUSTER (step4)...\n")

    # -------------------------------------------------------------------------
    # Profilazione cluster (IDENTICA AL TUO CODICE ORIGINALE)
    # -------------------------------------------------------------------------

    for k in clusters:
        sub = df[df["cluster"] == k].copy()
        n = len(sub)

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🔵 CLUSTER {k} — {n} partite\n")

        # 1X2
        p1 = pct(sub["result_1x2"] == "1")
        px = pct(sub["result_1x2"] == "X")
        p2 = pct(sub["result_1x2"] == "2")

        print("📌 Risultati reali 1X2:")
        print(f"   1: {p1:.2f}%")
        print(f"   X: {px:.2f}%")
        print(f"   2: {p2:.2f}%\n")

        # OU
        tg = sub["total_goals"]
        ou_15_over = pct(tg >= 2)
        ou_25_over = pct(tg >= 3)
        ou_35_over = pct(tg >= 4)

        print("📌 Over/Under:")
        print(f"   Over 1.5: {ou_15_over:.2f}%   |   Under 1.5: {100-ou_15_over:.2f}%")
        print(f"   Over 2.5: {ou_25_over:.2f}%   |   Under 2.5: {100-ou_25_over:.2f}%")
        print(f"   Over 3.5: {ou_35_over:.2f}%   |   Under 3.5: {100-ou_35_over:.2f}%\n")

        # GG/NG
        gg = pct((sub["home_ft"] > 0) & (sub["away_ft"] > 0))
        print("📌 GG / NG:")
        print(f"   GG: {gg:.2f}%")
        print(f"   NG: {100-gg:.2f}%\n")

        # Segna casa/ospite
        segna_casa = pct(sub["home_ft"] > 0)
        segna_osp = pct(sub["away_ft"] > 0)
        entrambe = pct((sub["home_ft"] > 0) & (sub["away_ft"] > 0))
        nessuna = pct((sub["home_ft"] == 0) & (sub["away_ft"] == 0))

        print("📌 Segna Casa / Ospite:")
        print(f"   Casa segna:     {segna_casa:.2f}%")
        print(f"   Ospite segna:   {segna_osp:.2f}%")
        print(f"   Entrambe:       {entrambe:.2f}%")
        print(f"   Nessuna:        {nessuna:.2f}%\n")

        # 0–0
        zero_zero = pct((sub["home_ft"] == 0) & (sub["away_ft"] == 0))
        print("📌 0–0:")
        print(f"   0–0: {zero_zero:.2f}%\n")

        # Big wins
        big_total = pct(sub["abs_diff"] >= 3)
        big_home = pct(sub["goal_diff"] >= 3)
        big_away = pct(sub["goal_diff"] <= -3)

        print("📌 Big Wins (diff ≥ 3):")
        print(f"   Totali: {big_total:.2f}%")
        print(f"   Casa:   {big_home:.2f}%")
        print(f"   Ospite: {big_away:.2f}%\n")

        # 1 goal
        one_goal = pct(sub["abs_diff"] == 1)
        print("📌 1 gol di scarto:")
        print(f"   1 gol: {one_goal:.2f}%\n")

        # MG
        mg_1_3 = pct((tg >= 1) & (tg <= 3))
        mg_1_4 = pct((tg >= 1) & (tg <= 4))
        mg_1_5 = pct((tg >= 1) & (tg <= 5))

        print("📌 MG totale:")
        print(f"   1–3: {mg_1_3:.2f}%")
        print(f"   1–4: {mg_1_4:.2f}%")
        print(f"   1–5: {mg_1_5:.2f}%\n")

        # Scoreline
        sub["score"] = sub["home_ft"].astype(int).astype(str) + "-" + sub["away_ft"].astype(int).astype(str)
        top_scores = sub["score"].value_counts(normalize=True).head(3)

        print("📌 Scoreline più frequenti:")
        for score, freq in top_scores.items():
            print(f"   {score}: {freq*100:.2f}%")

        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    print("🟢 Profilazione completata!")


if __name__ == "__main__":
    main()