from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.db.database_football import get_football_db
from app.db.models_football import Fixture
from app.ml.over_model.v1.model import OverModelV1
from app.ml.mg.v1.model import MGModelV1
from datetime import datetime
import json
import os

router = APIRouter()


@router.post("/generate")
async def generate_recommendations(save: bool = True) -> Dict[str, Any]:
    """
    Genera predizioni e raccomandazioni da tutti i modelli (Over, MG, ecc.)
    per tutte le fixtures.
    Se save=True salva in data/all_predictions.json.
    """
    try:
        db = next(get_football_db())
        try:
            fixtures = db.query(Fixture).all()

            # ===== Modelli attivi =====
            over_model = OverModelV1(debug=False)
            mg_model = MGModelV1(
                patch=0,
                calibrators_global_dir="app/ml/mg_model/v1/calibrators/global",
                calibrators_leagues_dir="app/ml/mg_model/v1/calibrators/leagues",
                use_league_calibrators=True
            )

            results = []

            for f in fixtures:
                # ======= OVER MODEL =======
                over_pred, over_recs = None, []
                # ======= OVER MODEL =======
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
                        if over_pred.get("p_over_0_5") >= 0.9:
                            over_recs.append({
                                "market": "Over 0.5",
                                "confidence": over_pred.get("p_over_0_5"),
                                "threshold": 0.9
                            })
                        if over_pred.get("p_over_1_5") >= 0.75:
                            over_recs.append({
                                "market": "Over 1.5",
                                "confidence": over_pred.get("p_over_1_5"),
                                "threshold": 0.75
                            })
                        if over_pred.get("p_over_2_5") >= 0.55:
                            over_recs.append({
                                "market": "Over 2.5",
                                "confidence": over_pred.get("p_over_2_5"),
                                "threshold": 0.55
                            })
                else:
                    # Se non ci sono quote Over/Under, ignora Over ma prosegui con MG
                    over_pred, over_recs = None, []
                        

                # ======= MG MODEL =======
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
                    mg_pred = mg_model.predict(mg_match)
                    if mg_pred and mg_pred.probs:
                        for market, prob in mg_pred.probs.items():
                            if prob is not None and prob >= 0.75:
                                mg_recs.append({
                                    "market": market,
                                    "confidence": prob,
                                    "threshold": 0.75
                                })

                # ======= COMBINA RACCOMANDAZIONI =======
                combined_recommendations = over_recs + mg_recs
                if not (over_pred or mg_pred):
                    continue

                results.append({
                    "fixture_id": str(f.id),
                    "match_date": f.match_date.isoformat() if f.match_date else None,
                    "home_team": f.home_team.name if f.home_team else None,
                    "away_team": f.away_team.name if f.away_team else None,
                    "league_code": f.league_code,
                    "odds": {
                        "home": f.avg_home_odds,
                        "draw": f.avg_draw_odds,
                        "away": f.avg_away_odds,
                        "over_25": f.avg_over_25_odds,
                        "under_25": f.avg_under_25_odds
                    },
                    "predictions": {
                        "over": over_pred,
                        "mg": mg_pred.probs if mg_pred else None
                    },
                    "recommendations": combined_recommendations
                })

            # ===== OUTPUT =====
            output = {
                "success": True,
                "fixtures_count": len(results),
                "generated_at": datetime.utcnow().isoformat(),
                "models": {
                    "over": "OverModelV1_patch7",
                    "mg": "MGModelV1_baseline"
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
