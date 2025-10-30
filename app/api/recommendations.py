from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.db.database_football import get_football_db
from app.db.models_football import Fixture
import json
import os
from app.ml.over_model.v1.model import OverModelV1
from datetime import datetime
router = APIRouter()


@router.post("/generate")
async def generate_over_recommendations(save: bool = True) -> Dict[str, Any]:
    """
    Genera raccomandazioni Over (0.5 / 1.5 / 2.5) su tutte le fixtures.
    Se save=True salva in data/all_over_recommendations.json.
    """
    try:
        db = next(get_football_db())
        try:
            fixtures = db.query(Fixture).all()
            model = OverModelV1(debug=False)

            results = []
            for f in fixtures:

                # skip fixtures senza odds
                if not f.avg_over_25_odds or not f.avg_under_25_odds:
                    continue

                match = {
                    "Avg>2.5": f.avg_over_25_odds,
                    "Avg<2.5": f.avg_under_25_odds,
                    "league_code": f.league_code,
                }
                pred = model.predict(match)
                recs = model.recommend(pred) if pred else []

                if not pred:
                    continue

                """  markets = [
                    ("Over 0.5 Goals", pred.get("p_over_0_5")),
                    ("Over 1.5 Goals", pred.get("p_over_1_5")),
                    ("Over 2.5 Goals", pred.get("p_over_2_5")),
                ]

                # ordina per probabilità decrescente
                markets = sorted(
                    [(m, p) for m, p in markets if p is not None],
                    key=lambda x: x[1],
                    reverse=True
                )

                # prendi solo i primi due mercati "migliori"
                recommendations = []
                for m, p in markets[:2]:
                    suggest = p > 0.60  # 60% soglia base
                    # EV solo per Over 2.5
                    ev = pred.get("ev_over_2_5") if m == "Over 2.5 Goals" else None
                    if m == "Over 2.5 Goals":
                        suggest = pred.get("suggest_over_2_5", False)

                    recommendations.append({
                        "market": m,
                        "confidence": round(p * 100, 1),
                        "ev": ev,
                        "suggest": suggest
                    }) """

                results.append({
                    "fixture_id": str(f.id),
                    "match_date": f.match_date.isoformat() if f.match_date else None,
                    "home_team": f.home_team.name if f.home_team else None,
                    "away_team": f.away_team.name if f.away_team else None,
                    "league_code": f.league_code,
                    "odds": {
                        "over_25": f.avg_over_25_odds,
                        "under_25": f.avg_under_25_odds
                    },
                    "predictions": pred,
                    "recommendations": recs
                })

            output = {
                "success": True,
                "fixtures_count": len(results),
                "generated_at": datetime.utcnow().isoformat(),
                "model": "OverModelV1_patch7",
                "fixtures": results
            }

            if save:
                os.makedirs("data", exist_ok=True)
                path = "data/all_over_recommendations.json"
                with open(path, "w", encoding="utf-8") as fjson:
                    json.dump(output, fjson, indent=2, default=str)
            
            return output

        finally:
            db.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
