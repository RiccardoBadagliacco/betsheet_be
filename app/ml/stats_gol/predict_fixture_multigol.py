# ==============================================
# app/ml/stats_gol/predict_fixtures_batch.py
# ==============================================

import pandas as pd
from sqlalchemy import or_
from sqlalchemy.orm import aliased
from tqdm import tqdm
from datetime import datetime

from app.db.database_football import FootballSessionLocal
from app.db.models import Fixture, Season, League, Team
from app.ml.stats_gol.model import MgFavModelV1, FAV_HOME_THR, FAV_AWAY_THR


def load_future_fixtures(db):
    """Carica tutte le fixture future con filtri per leghe e squadre top."""
    TARGET_LEAGUES = ["I1", "E0", "SP1", "D1", "DKN"]
    TOP_TEAMS = [
        "Inter", "Milan", "Roma", "Juventus", "Napoli",
        "Chelsea", "Manchester City", "Liverpool", "Arsenal",
        "Real Madrid", "Ath Madrid", "Barcellona",
        "Bayern Munich", "Dortmund", "Leverkusen", "FC Copenhagen"
    ]

    home_team = aliased(Team)
    away_team = aliased(Team)

    q = (
        db.query(Fixture)
        .join(Season, Fixture.season_id == Season.id, isouter=True)
        .join(League, Fixture.league_code == League.code, isouter=True)
        .join(home_team, Fixture.home_team_id == home_team.id, isouter=True)
        .join(away_team, Fixture.away_team_id == away_team.id, isouter=True)
        .filter(
            or_(
                home_team.name.in_(TOP_TEAMS),
                away_team.name.in_(TOP_TEAMS),
            ),
            Fixture.league_code.in_(TARGET_LEAGUES),
            or_(
                Fixture.avg_home_odds < FAV_HOME_THR,
                Fixture.avg_away_odds < FAV_AWAY_THR,
            ),
        )
        .order_by(Fixture.match_date.asc())
    )
    return q.all()


def print_header():
    print("\n=== 🔮 PREVISIONI MULTIGOL PER FIXTURE FUTURE ===\n")
    print(f"{'Data':<12} {'Lega':<4} {'Partita':<35} {'Fav.':<6} {'λ':>5}   "
          f"{'MG 1-3':>8} {'MG 1-4':>8} {'MG 1-5':>8}")
    print("-" * 90)


def print_row(row):
    match = f"{row['home_team']} - {row['away_team']}"
    print(f"{row['match_date']:<12} {row['league']:<4} {match:<35} "
          f"{row['favorite_side']:<6} {row['lambda']:>5.2f}   "
          f"{row['MG_1_3']:>7.1f}% {row['MG_1_4']:>7.1f}% {row['MG_1_5']:>7.1f}%")


if __name__ == "__main__":
    db = FootballSessionLocal()
    try:
        fixtures = load_future_fixtures(db)
        if not fixtures:
            print("⚠️ Nessuna fixture trovata con i filtri specificati.")
            exit()

        model = MgFavModelV1(db,debug=True)
        results = []

        print(f"\nElaborazione di {len(fixtures)} fixture future...\n")
        print_header()

        for f in tqdm(fixtures, total=len(fixtures), ncols=80, colour="cyan"):
            pred = model.predict_for_match(f)
            if not pred:
                continue

            fav = pred["favorite_side"]
            row = {
                "fixture_id": str(f.id),
                "match_date": f.match_date.strftime("%Y-%m-%d") if f.match_date else None,
                "league": f.league_code,
                "home_team": f.home_team.name if f.home_team else None,
                "away_team": f.away_team.name if f.away_team else None,
                "favorite_side": fav,
            }

            if fav == "home":
                row.update({
                    "lambda": round(pred.get("lambda_home", 0), 3),
                    "MG_1_3": round(pred.get("MG_Casa_1_3", 0)*100, 1),
                    "MG_1_4": round(pred.get("MG_Casa_1_4", 0)*100, 1),
                    "MG_1_5": round(pred.get("MG_Casa_1_5", 0)*100, 1),
                })
            elif fav == "away":
                row.update({
                    "lambda": round(pred.get("lambda_away", 0), 3),
                    "MG_1_3": round(pred.get("MG_Ospite_1_3", 0)*100, 1),
                    "MG_1_4": round(pred.get("MG_Ospite_1_4", 0)*100, 1),
                    "MG_1_5": round(pred.get("MG_Ospite_1_5", 0)*100, 1),
                })

            results.append(row)
            print_row(row)

        if not results:
            print("\n⚠️ Nessuna fixture valida elaborata (quote o dati mancanti).")
            exit()

        df = pd.DataFrame(results)
        df.to_csv("data/mg_fixtures_predictions.csv", index=False)

        print("\n" + "-" * 90)
        print(f"✅ Totale fixture elaborate: {len(df)}")
        print(f"💾 Risultati salvati in: data/mg_fixtures_predictions.csv\n")

    finally:
        db.close()
