from fastapi import APIRouter, Query
from typing import Optional
from addons.compute_team_profiles import compute_team_profiles
import os
import json

router = APIRouter()

@router.get("/team-profiles")
def get_team_profiles(
    rolling_n: int = Query(15, description="Numero di partite recenti da considerare"),
    season: Optional[str] = Query(None, description="Filtra per stagione, es. '2025/26'"),
    use_weighted: bool = Query(True, description="Usa media pesata esponenziale sulle ultime partite")
):
    """
    Restituisce i profili delle squadre calcolati dal database e li salva in data/team_profiles.json.
    """
    profiles = compute_team_profiles(rolling_n=rolling_n, season_filter=season, use_weighted=use_weighted)
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "team_profiles.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)
    return {"success": True, "profiles": profiles, "count": len(profiles), "json_file": out_path}
