#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5 — SOFT ENGINE OFFLINE EVAL

Obiettivo:
- Per ogni match storico:
    * calcolare le soft-probabilities usando il soft_engine_api_v2
    * agganciare il risultato reale (home_ft, away_ft)
- Salvare tutto in:
    data/step5_soft_history.parquet

Con questa tabella potrai poi fare query tipo:
    - "tra i match con soft_pO25 >= 0.60, quante volte finisce Over 1.5?"
    - "se soft_p1 >= a e soft_pO15 >= b, quante volte finisce GG?"
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# -----------------------
# PATH & IMPORT
# -----------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR   = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR   = AFFINI_DIR / "data"

SLIM_FILE  = DATA_DIR / "step4b_affini_index_slim_v2.parquet"
WIDE_FILE  = DATA_DIR / "step4a_affini_index_wide_v2.parquet"
OUT_FILE   = DATA_DIR / "step5_soft_history.parquet"

# importa il soft engine (adatta l'import se il modulo è altrove)
from app.ml.correlazioni_affini_v2.common.soft_engine_api_v2 import run_soft_engine_api


def main():
    slim = pd.read_parquet(SLIM_FILE)
    wide = pd.read_parquet(WIDE_FILE)

    # Consideriamo solo match con risultato noto
    wide_played = wide.dropna(subset=["home_ft", "away_ft"]).copy()
    played_ids = set(wide_played["match_id"].unique())

    slim_played = slim[slim["match_id"].isin(played_ids)].copy()

    rows = []

    # opzionale: puoi limitare il numero di match per debug
    # slim_played = slim_played.sample(5000, random_state=42)

    for _, row in tqdm(
        slim_played.iterrows(),
        total=len(slim_played),
        desc="Processing matches",
        unit="match",
    ):
        mid = row["match_id"]

        res = run_soft_engine_api(
            target_match_id=mid,
            slim=slim,
            wide=wide,
            top_n=80,
            min_neighbors=30,
        )

        if res.get("status") != "ok":
            # salviamo comunque info minime, così sappiamo dove fallisce
            rows.append({
                "match_id": mid,
                "status": res.get("status"),
                "reason": res.get("reason"),
            })
            continue

        soft = res["soft_probs"]

        # esito reale
        wrow = wide_played.loc[wide_played["match_id"] == mid].iloc[0]
        gh = int(wrow["home_ft"])
        ga = int(wrow["away_ft"])
        tg = gh + ga

        is_over15 = int(tg >= 2)
        is_over25 = int(tg >= 3)
        is_gg     = int((gh > 0) and (ga > 0))

        rows.append({
            "match_id": mid,
            "status": "ok",

            "date": wrow.get("date"),
            "league": wrow.get("league"),
            "home_team": wrow.get("home_team"),
            "away_team": wrow.get("away_team"),

            # cluster di riferimento
            "cluster_1x2": wrow.get("cluster_1x2"),
            "cluster_ou25": wrow.get("cluster_ou25"),
            "cluster_ou15": wrow.get("cluster_ou15"),

            # soft probabilities
            "soft_p1":  soft["p1"],
            "soft_px":  soft["px"],
            "soft_p2":  soft["p2"],
            "soft_pO15": soft["pO15"],
            "soft_pU15": soft["pU15"],
            "soft_pO25": soft["pO25"],
            "soft_pU25": soft["pU25"],

            # esito reale
            "home_ft": gh,
            "away_ft": ga,
            "total_goals": tg,
            "is_over15_real": is_over15,
            "is_over25_real": is_over25,
            "is_gg_real": is_gg,
        })

    df_soft = pd.DataFrame(rows)
    # tieni solo quelli andati a buon fine
    df_soft_ok = df_soft[df_soft["status"] == "ok"].copy()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_soft_ok.to_parquet(OUT_FILE, index=False)
    print(f"💾 Salvato soft_history in: {OUT_FILE}")


if __name__ == "__main__":
    main()
