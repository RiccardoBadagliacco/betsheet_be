# ============================================================
# app/ml/profeta/step3_profeta_batch.py
# ============================================================

"""
STEP3 — Batch predictions PROFETA V0

Obiettivo:
    Applicare ProfetaEngine a TUTTE le righe di step0_profeta.parquet
    e salvare le probabilità dei mercati in un parquet:

    data/step3_profeta_predictions.parquet
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR / "data"
STEP0_PATH = DATA_DIR / "step0_profeta.parquet"
STEP3_OUT  = DATA_DIR / "step3_profeta_predictions.parquet"

# Importa l'engine dallo step2 (versione v3)
from app.ml.profeta.step2_profeta import ProfetaEngine


def build_step3_predictions():
    print("📥 Carico step0_profeta:", STEP0_PATH)
    df = pd.read_parquet(STEP0_PATH)
    print("   → righe totali:", len(df))

    engine = ProfetaEngine(max_goals=10)

    rows_out = []

    for idx, row in df.iterrows():
        res = engine.predict_from_row(row)
        markets = res["markets"]

        out_row = {
            "match_id": row["match_id"],
            "is_fixture": row["is_fixture"],
            "league_id": row["league_id"],
            "season_id": row["season_id"],
            "home_team_id": row["home_team_id"],
            "away_team_id": row["away_team_id"],
            "lambda_home": markets["lambda_home"],
            "lambda_away": markets["lambda_away"],
            "xg_total": markets["xg_total"],

            # 1X2
            "p1": markets["p1"],
            "px": markets["px"],
            "p2": markets["p2"],

            # GG / NoGG
            "p_gg": markets["p_gg"],
            "p_nogg": markets["p_nogg"],

            # O/U
            "p_over_1_5": markets["p_over_1_5"],
            "p_under_1_5": markets["p_under_1_5"],
            "p_over_2_5": markets["p_over_2_5"],
            "p_under_2_5": markets["p_under_2_5"],
            "p_over_3_5": markets["p_over_3_5"],
            "p_under_3_5": markets["p_under_3_5"],

            # MG totali
            "p_mg_1_3": markets["p_mg_1_3"],
            "p_mg_1_4": markets["p_mg_1_4"],
            "p_mg_1_5": markets["p_mg_1_5"],
            "p_mg_2_4": markets["p_mg_2_4"],
            "p_mg_2_5": markets["p_mg_2_5"],
            "p_mg_2_6": markets["p_mg_2_6"],

            # MG casa
            "p_mg_home_1_3": markets["p_mg_home_1_3"],
            "p_mg_home_1_4": markets["p_mg_home_1_4"],
            "p_mg_home_1_5": markets["p_mg_home_1_5"],

            # MG away
            "p_mg_away_1_3": markets["p_mg_away_1_3"],
            "p_mg_away_1_4": markets["p_mg_away_1_4"],
            "p_mg_away_1_5": markets["p_mg_away_1_5"],

            # extra
            "p_0_0": markets["p_0_0"],
            "p_home_clean_sheet": markets["p_home_clean_sheet"],
            "p_away_clean_sheet": markets["p_away_clean_sheet"],
            "top6_correct_score": json.dumps(markets["top6_correct_score"]),
        }

        rows_out.append(out_row)

        if (idx + 1) % 1000 == 0:
            print(f"   → processate {idx+1} righe…")

    df_out = pd.DataFrame(rows_out)
    print("📊 shape finale predictions:", df_out.shape)

    df_out.to_parquet(STEP3_OUT, index=False)
    print("💾 Salvato:", STEP3_OUT)


def main():
    print("🚀 STEP3 PROFETA — Batch predictions")
    build_step3_predictions()
    print("✅ STEP3 completato.")


if __name__ == "__main__":
    main()
