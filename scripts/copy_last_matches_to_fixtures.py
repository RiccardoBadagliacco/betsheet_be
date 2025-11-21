"""
Utility per copiare le ultime N partite concluse in tabella fixtures
senza riportare i risultati (goal a/b restano NULL).
"""

from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.database_football import get_football_db
from app.db.models_football import Match, Fixture


def copy_latest_matches_to_fixtures(limit: int = 4) -> None:
    db: Session = next(get_football_db())
    try:
        matches = (
            db.query(Match)
            .order_by(desc(Match.match_date), desc(Match.match_time))
            .limit(limit)
            .all()
        )

        created = 0
        for m in matches:
            # evita doppioni sulla stessa data/squadre
            exists = (
                db.query(Fixture)
                .filter(
                    Fixture.match_date == m.match_date,
                    Fixture.home_team_id == m.home_team_id,
                    Fixture.away_team_id == m.away_team_id,
                )
                .first()
            )
            if exists:
                continue

            league = m.season.league if m.season else None
            fixture = Fixture(
                season_id=m.season_id,
                home_team_id=m.home_team_id,
                away_team_id=m.away_team_id,
                match_date=m.match_date,
                match_time=m.match_time,
                avg_home_odds=m.avg_home_odds,
                avg_draw_odds=m.avg_draw_odds,
                avg_away_odds=m.avg_away_odds,
                avg_over_25_odds=m.avg_over_25_odds,
                avg_under_25_odds=m.avg_under_25_odds,
                league_code=getattr(league, "code", None),
                league_name=getattr(league, "name", None),
                # risultati/ statistiche non copiati
                home_goals_ft=None,
                away_goals_ft=None,
                home_goals_ht=None,
                away_goals_ht=None,
                home_shots=None,
                away_shots=None,
                home_shots_target=None,
                away_shots_target=None,
            )
            db.add(fixture)
            created += 1

        db.commit()
        print(f"✅ Copiate {created} fixture su {len(matches)} match analizzati.")
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    copy_latest_matches_to_fixtures()
