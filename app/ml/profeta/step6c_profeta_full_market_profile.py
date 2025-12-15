"""
STEP6C — PROFETA FULL MARKET PROFILE

Crea la mappa:
CONTROL_STATE × GOAL_STATE → MERCATI REALI

Serve per:
- capire quali mercati hanno struttura
- sapere cosa ha senso guardare a runtime
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

STEP0 = DATA_DIR / "step0_profeta.parquet"
STEP5A = DATA_DIR / "step5a_profeta_control_state.parquet"
STEP5B = DATA_DIR / "step5b_profeta_goal_state.parquet"
OUT = DATA_DIR / "step6c_profeta_full_market_profile.parquet"


def main():
    print("🚀 STEP6C — FULL MARKET PROFILE")

    df0 = pd.read_parquet(STEP0)
    df_c = pd.read_parquet(STEP5A)
    df_g = pd.read_parquet(STEP5B)

    df = (
        df0.merge(df_c, on="match_id")
           .merge(df_g, on="match_id")
    )

    # solo storico
    df = df[df["is_fixture"] == False].copy()
    print(f"📦 Partite storiche: {len(df)}")

    # =========================
    # EVENTI REALI
    # =========================
    df["HOME_WIN"] = df["home_goals"] > df["away_goals"]
    df["DRAW"] = df["home_goals"] == df["away_goals"]
    df["AWAY_WIN"] = df["home_goals"] < df["away_goals"]

    df["OVER15"] = (df["home_goals"] + df["away_goals"]) >= 2
    df["OVER25"] = (df["home_goals"] + df["away_goals"]) >= 3
    df["OVER35"] = (df["home_goals"] + df["away_goals"]) >= 4
    df["UNDER25"] = (df["home_goals"] + df["away_goals"]) <= 2

    def mg(a, b, x):
        return (x >= a) & (x <= b)

    df["MG_1_3"] = mg(1, 3, df["home_goals"] + df["away_goals"])
    df["MG_1_4"] = mg(1, 4, df["home_goals"] + df["away_goals"])
    df["MG_1_5"] = mg(1, 5, df["home_goals"] + df["away_goals"])
    df["MG_2_4"] = mg(2, 4, df["home_goals"] + df["away_goals"])
    df["MG_2_5"] = mg(2, 5, df["home_goals"] + df["away_goals"])

    df["MG_HOME_1_3"] = mg(1, 3, df["home_goals"])
    df["MG_HOME_1_4"] = mg(1, 4, df["home_goals"])
    df["MG_HOME_1_5"] = mg(1, 5, df["home_goals"])

    df["MG_AWAY_1_3"] = mg(1, 3, df["away_goals"])
    df["MG_AWAY_1_4"] = mg(1, 4, df["away_goals"])
    df["MG_AWAY_1_5"] = mg(1, 5, df["away_goals"])

    # =========================
    # PROFILO
    # =========================
    markets = [
        "HOME_WIN", "DRAW", "AWAY_WIN",
        "OVER15", "OVER25", "OVER35", "UNDER25",
        "MG_1_3", "MG_1_4", "MG_1_5", "MG_2_4", "MG_2_5",
        "MG_HOME_1_3", "MG_HOME_1_4", "MG_HOME_1_5",
        "MG_AWAY_1_3", "MG_AWAY_1_4", "MG_AWAY_1_5",
    ]

    rows = []
    for (cs, gs), g in df.groupby(["control_state", "goal_state"]):
        if len(g) < 300:
            continue

        row = {
            "control_state": cs,
            "goal_state": gs,
            "support": len(g),
            "support_pct": len(g) / len(df),
        }

        for m in markets:
            row[m] = g[m].mean()

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("support", ascending=False)
    out.to_parquet(OUT, index=False)

    print(f"💾 Salvato: {OUT}")
    print("✅ STEP6C COMPLETATO")


if __name__ == "__main__":
    main()