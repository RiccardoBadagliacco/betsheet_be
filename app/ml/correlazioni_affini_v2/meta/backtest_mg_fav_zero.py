#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crea un parquet contenente SOLO le partite in cui la favorita ha quota <= 1.95,
salva le quote 1X2 per ogni partita e aggiunge:

- mg_fav_1_3
- mg_fav_1_4
- mg_fav_1_5
- score1, score2, score3, score4  (scoreline più probabili)

calcolati dal motore stepZ.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from sqlalchemy.orm import joinedload
from tqdm import tqdm

from app.db.database_football import get_football_db_session
from app.db import models as db_models

from app.ml.correlazioni_affini_v2.meta.stepZ_decision_engine import run_decision
from app.ml.correlazioni_affini_v2.meta.stepZ_formatter import build_final_forecast

# ------------------------------------------------------------
# OUTPUT PATH
# ------------------------------------------------------------

OUT_DIR = REPO_ROOT / "app" / "ml" / "backtests" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQUET = OUT_DIR / "matches_fav_le_195.parquet"


# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------

def _get_favourite_side(m: db_models.Match) -> Optional[Dict[str, Any]]:
    """
    Determina la favorita tra home e away.
    Ignora la X.
    """
    h = m.avg_home_odds
    a = m.avg_away_odds

    if h is None or a is None:
        return None

    if h <= a:
        return {"side": "home", "odds": h}
    else:
        return {"side": "away", "odds": a}


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    print("📡 Connessione al DB football...")
    db_gen = get_football_db_session()
    db = next(db_gen)

    print("📥 Recupero match con risultato...")
    q = (
        db.query(db_models.Match)
        .options(
            joinedload(db_models.Match.season).joinedload(db_models.Season.league),
            joinedload(db_models.Match.home_team),
            joinedload(db_models.Match.away_team),
        )
        .filter(
            db_models.Match.home_goals_ft.isnot(None),
            db_models.Match.away_goals_ft.isnot(None),
        )
    )

    matches = q.all()
    print(f"Totale match storici con risultato: {len(matches)}")

    records: List[Dict[str, Any]] = []

    for m in tqdm(matches, desc="Elaboro match", unit="match"):

        fav = _get_favourite_side(m)
        if not fav:
            continue

        fav_side = fav["side"]
        fav_odds = fav["odds"]

        # ⭐ Filtro richiesta: favorita <= 1.95
        if fav_odds is None or fav_odds > 1.95:
            continue

        league = m.season.league if m.season else None

        mg_fav_1_3 = mg_fav_1_4 = mg_fav_1_5 = None
        score1 = score2 = score3 = score4 = None

        match_id_str = str(m.id)

        # ---- Calcolo MG favorita + scoreline tramite stepZ
        try:
            raw = run_decision(match_id_str)
            final = build_final_forecast(raw)

            # MG favorita
            mg_fav = (final.get("multigol") or {}).get("fav") or {}
            mg_fav_1_3 = float(mg_fav.get("1-3")) if "1-3" in mg_fav else None
            mg_fav_1_4 = float(mg_fav.get("1-4")) if "1-4" in mg_fav else None
            mg_fav_1_5 = float(mg_fav.get("1-5")) if "1-5" in mg_fav else None

            # Scoreline
            score_pred = final.get("score_prediction") or []
            if len(score_pred) > 0:
                score1 = score_pred[0]["score"]
            if len(score_pred) > 1:
                score2 = score_pred[1]["score"]
            if len(score_pred) > 2:
                score3 = score_pred[2]["score"]
            if len(score_pred) > 3:
                score4 = score_pred[3]["score"]

        except Exception:
            pass

        records.append({
            "match_id": match_id_str,
            "match_date": m.match_date,
            "league_code": league.code if league else None,

            # Teams + risultati
            "home_team": m.home_team.name if m.home_team else None,
            "away_team": m.away_team.name if m.away_team else None,
            "home_ft": m.home_goals_ft,
            "away_ft": m.away_goals_ft,

            # Quote 1X2
            "avg_home_odds": m.avg_home_odds,
            "avg_draw_odds": m.avg_draw_odds,
            "avg_away_odds": m.avg_away_odds,

            # Favorita
            "fav_side": fav_side,
            "fav_odds": fav_odds,

            # Multigol favorita
            "mg_fav_1_3": mg_fav_1_3,
            "mg_fav_1_4": mg_fav_1_4,
            "mg_fav_1_5": mg_fav_1_5,

            # Scoreline principali
            "score1": score1,
            "score2": score2,
            "score3": score3,
            "score4": score4,
        })

    df = pd.DataFrame(records)
    print(f"\nMatch che rispettano favorita <= 1.95: {len(df)}")

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"💾 File salvato: {OUT_PARQUET}")


if __name__ == "__main__":
    main()