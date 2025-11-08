from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.db.database_football import get_football_db
from app.db.models_football import Fixture, League, Season
from app.ml.over_model.v1.model import OverModelV1
from app.ml.mg.v1.model import MGModelV1
from app.ml.stats_gol.model import MgFavModelV1  # 👈 nuovo modello statistico
from datetime import datetime
from sqlalchemy.orm import joinedload
import json
import os

router = APIRouter()


@router.post("/generate")
async def generate_recommendations(save: bool = True) -> Dict[str, Any]:
    """
    Genera predizioni e raccomandazioni da tutti i modelli (Over, MG AI, MG Stat)
    per tutte le fixtures future.
    Salva in data/all_predictions.json se save=True.
    """
    try:
        db = next(get_football_db())
        try:
            # Carica le fixtures con le relazioni necessarie per evitare query N+1
            fixtures = db.query(Fixture).options(
                joinedload(Fixture.season).joinedload(Season.league),
                joinedload(Fixture.home_team),
                joinedload(Fixture.away_team)
            ).all()
            results = []

            # ===== Inizializza modelli =====
            over_model = OverModelV1(debug=False)
            mg_ai_model = MGModelV1(
                patch=0,
                calibrators_global_dir="app/ml/mg_model/v1/calibrators/global",
                calibrators_leagues_dir="app/ml/mg_model/v1/calibrators/leagues",
                use_league_calibrators=True
            )
            mg_stat_model = MgFavModelV1(db, window=30, debug=False)  # 👈 nuovo modello

            for f in fixtures:
                # ======= OVER/UNDER MODEL =======
                over_pred, over_recs = None, []
                if f.avg_over_25_odds and f.avg_under_25_odds:
                    over_match = {
                        "Avg>2.5": f.avg_over_25_odds,
                        "Avg<2.5": f.avg_under_25_odds,
                        "league_code": f.league_code,
                    }
                    over_pred = over_model.predict(over_match)
                    if over_pred:
                        over_recs = []
                        for market, threshold in [
                            ("Over 0.5", 0.9),
                            ("Over 1.5", 0.75),
                            ("Over 2.5", 0.55)
                        ]:
                            # Il modello restituisce chiavi con underscore: p_over_0_5, p_over_1_5, p_over_2_5
                            prob_key = f"p_{market.lower().replace(' ', '_').replace('.', '_')}"
                            if over_pred.get(prob_key) and over_pred[prob_key] >= threshold:
                                over_recs.append({
                                    "market": market,
                                    "confidence": over_pred[prob_key],
                                    "threshold": threshold
                                })

                # ======= AI MULTIGOL MODEL (vecchio) =======
                mg_pred, mg_recs = None, []
                if f.avg_home_odds and f.avg_draw_odds and f.avg_away_odds:
                    mg_match = {
                        "avg_home_odds": f.avg_home_odds,
                        "avg_draw_odds": f.avg_draw_odds,
                        "avg_away_odds": f.avg_away_odds,
                        "avg_over_25_odds": f.avg_over_25_odds,
                        "avg_under_25_odds": f.avg_under_25_odds,
                        "league_code": f.league_code,
                        "season": f.season.name if f.season else None,
                    }
                    mg_pred = mg_ai_model.predict(mg_match)
                    if mg_pred and mg_pred.probs:
                        for market, prob in mg_pred.probs.items():
                            if prob is not None and prob >= 0.75:
                                mg_recs.append({
                                    "market": market,
                                    "confidence": prob,
                                    "threshold": 0.75
                                })

                # ======= STATISTICAL MULTIGOL MODEL (nuovo) =======
                mgstat_pred, mgstat_recs = None, []
                try:
                    mgstat_pred = mg_stat_model.predict_for_match(f)  # o predict_for_fixture alias
                    if mgstat_pred:
                        fav = mgstat_pred.get("favorite_side")
                        if fav:
                            # Standardizza le label: MG_HOME_1_3, MG_AWAY_1_3, etc.
                            prefix = "MG_HOME" if fav == "home" else "MG_AWAY"
                            for mg_name in ["1_3", "1_4", "1_5"]:
                                key = f"{prefix}_{mg_name}"
                                prob = mgstat_pred.get(key)
                                if prob is not None:
                                    mgstat_recs.append({
                                        "market": key,  # Standardizzato: MG_HOME_1_3, MG_AWAY_1_4, etc.
                                        "confidence": round(prob, 3),
                                        "threshold": 75
                                    })
                except Exception as e:
                    print(f"[MG_STAT ERROR] {e}")
                    pass

                # ======= AGGREGAZIONE =======
                if not (over_recs or mg_recs or mgstat_recs):
                    continue

                results.append({
                    "fixture_id": str(f.id),
                    "match_date": f.match_date.isoformat() if f.match_date else None,
                    "home_team": f.home_team.name if f.home_team else None,
                    "away_team": f.away_team.name if f.away_team else None,
                    "league_code": f.league_code,
                    "is_favorite_league": f.season.league.is_favorite if (f.season and f.season.league) else False,
                    "home_team_is_top": f.home_team.is_top if f.home_team else False,
                    "away_team_is_top": f.away_team.is_top if f.away_team else False,
                    "odds": {
                        "home": f.avg_home_odds,
                        "draw": f.avg_draw_odds,
                        "away": f.avg_away_odds,
                        "over_25": f.avg_over_25_odds,
                        "under_25": f.avg_under_25_odds
                    },
                    "predictions": {
                        "over_under": over_pred,
                        "ai_multigol": mg_pred.probs if mg_pred else None,
                        "stat_multigol": mgstat_pred
                    },
                    "recommendations": {
                        "over_under": over_recs,
                        "ai_multigol": mg_recs,
                        "stat_multigol": mgstat_recs
                    }
                })

            # ===== OUTPUT =====
            output = {
                "success": True,
                "fixtures_count": len(results),
                "generated_at": datetime.utcnow().isoformat(),
                "models": {
                    "over_under": "OverModelV1_patch7",
                    "ai_multigol": "MGModelV1_baseline",
                    "stat_multigol": "MgFavModelV1_window30_decay"
                },
                "fixtures": results
            }

            if save:
                os.makedirs("data", exist_ok=True)
                path = "data/all_predictions.json"
                with open(path, "w", encoding="utf-8") as fjson:
                    json.dump(output, fjson, indent=2, default=str)

            return output

        finally:
            db.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
