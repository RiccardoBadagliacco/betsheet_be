# ==============================================
# app/ml/stats_gol/backtest.py
# ==============================================

import pandas as pd
import numpy as np
from tqdm import tqdm
from sqlalchemy import or_
from sqlalchemy.orm import aliased

from app.db.database_football import FootballSessionLocal
from app.db.models import Match, Season, League, Team, Fixture
from .model import MgFavModelV1, FAV_HOME_THR, FAV_AWAY_THR


# =======================================================
# CONFIG
# =======================================================
USE_FIXTURES = True  # True = fixture future, False = storico

TARGET_LEAGUES = ["I1", "E0", "SP1", "D1", "DKN","F1", "T1"]
TOP_TEAMS = [
    "Inter", "Milan", "Roma", "Juventus", "Napoli",
    "Chelsea", "Manchester City", "Liverpool", "Arsenal",
    "Real Madrid", "Ath Madrid", "Barcellona",
    "Bayern Munich", "Dortmund", "Leverkusen, Copenhagen","Paris SG","Marseille", "Galatasaray", "Fenerbahce"
]

def brier(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def load_future_fixtures(db):

    home_team = aliased(Team)
    away_team = aliased(Team)

    q = (
        db.query(Fixture)
        .join(Season, Fixture.season_id == Season.id, isouter=True)
        .join(League, Fixture.league_code == League.code, isouter=True)
        .join(home_team, Fixture.home_team_id == home_team.id, isouter=True)
        .join(away_team, Fixture.away_team_id == away_team.id, isouter=True)
        .filter(
            Fixture.league_code.in_(TARGET_LEAGUES),
            or_(
                home_team.name.in_(TOP_TEAMS),
                away_team.name.in_(TOP_TEAMS),
            ),
            or_(
                Fixture.avg_home_odds < FAV_HOME_THR,
                Fixture.avg_away_odds < FAV_AWAY_THR,
            ),
        )
        .order_by(Fixture.match_date.asc())
    )
    return q.all()


def load_current_season_matches(db):
    home_team = aliased(Team)
    away_team = aliased(Team)

    q = (
        db.query(Match)
        .join(Season, Match.season_id == Season.id)
        .join(League, Season.league_id == League.id)
        .join(home_team, Match.home_team_id == home_team.id)
        .join(away_team, Match.away_team_id == away_team.id)
        .filter(
            Match.home_goals_ft != None,
            Match.away_goals_ft != None,
            League.code.in_(TARGET_LEAGUES),
            or_(
                Match.avg_home_odds < FAV_HOME_THR,
                Match.avg_away_odds < FAV_AWAY_THR,
            ),
            or_(
                home_team.name.in_(TOP_TEAMS),
                away_team.name.in_(TOP_TEAMS),
            ),
        )
        .order_by(Match.match_date.asc())
    )
    return q.all()


# =======================================================
# MAIN
# =======================================================
if __name__ == "__main__":
    db = FootballSessionLocal()
    try:
        model = MgFavModelV1(db, window=20, debug=False)
        matches = load_future_fixtures(db) if USE_FIXTURES else load_current_season_matches(db)

        print(f"\n=== MODALITÀ: {'FIXTURES (LIVE)' if USE_FIXTURES else 'BACKTEST (STORICO)'} ===\n")

        rows = []
        for m in tqdm(matches, total=len(matches)):
            pred = model.predict_for_match(m)
            if not pred:
                continue

            fav = pred["favorite_side"]

            if USE_FIXTURES:
                rows.append({
                    "league": getattr(m, "league_code", None),
                    "fav_side": fav,
                    "home_team": m.home_team.name if m.home_team else None,
                    "away_team": m.away_team.name if m.away_team else None,
                    "λ": pred.get("lambda_home") or pred.get("lambda_away"),
                    "MG1_3": pred.get("MG_Casa_1_3") or pred.get("MG_Ospite_1_3"),
                    "MG1_4": pred.get("MG_Casa_1_4") or pred.get("MG_Ospite_1_4"),
                    "MG1_5": pred.get("MG_Casa_1_5") or pred.get("MG_Ospite_1_5"),
                })
            else:
                hg, ag = int(m.home_goals_ft), int(m.away_goals_ft)
                if fav == "home":
                    rows.append({
                        "league": getattr(m.season.league, "code", None),
                        "fav_side": "home",
                        "home_team": m.home_team.name,
                        "away_team": m.away_team.name,
                        "p1_3": pred["MG_Casa_1_3"],
                        "p1_4": pred["MG_Casa_1_4"],
                        "p1_5": pred["MG_Casa_1_5"],
                        "y1_3": int(1 <= hg <= 3),
                        "y1_4": int(1 <= hg <= 4),
                        "y1_5": int(1 <= hg <= 5),
                    })
                else:
                    rows.append({
                        "league": getattr(m.season.league, "code", None),
                        "fav_side": "away",
                        "home_team": m.home_team.name,
                        "away_team": m.away_team.name,
                        "p1_3": pred["MG_Ospite_1_3"],
                        "p1_4": pred["MG_Ospite_1_4"],
                        "p1_5": pred["MG_Ospite_1_5"],
                        "y1_3": int(1 <= ag <= 3),
                        "y1_4": int(1 <= ag <= 4),
                        "y1_5": int(1 <= ag <= 5),
                    })

        df = pd.DataFrame(rows)

        if USE_FIXTURES:
            print("\n=== 🔮 PREVISIONI MULTIGOL (FIXTURE FUTURE) ===")
            print(f"{'Lega':<4} {'Partita':<35} {'Fav.':<6} {'λ':>5}   {'MG 1-3':>8} {'MG 1-4':>8} {'MG 1-5':>8}")
            print("-" * 90)
            for _, r in df.iterrows():
                print(f"{r['league']:<4} {(r['home_team'] or '') + ' - ' + (r['away_team'] or ''):<35} "
                      f"{r['fav_side']:<6} {r['λ']:>5.2f}   {r['MG1_3']*100:>7.1f}% {r['MG1_4']*100:>7.1f}% {r['MG1_5']*100:>7.1f}%")

            # --- ANALISI MEDIA PER SQUADRA ---
            print("\n=== 📊 STATISTICHE PER SQUADRA (FIXTURE FUTURE) ===")

            for team in TOP_TEAMS:
                sub = df[(df["home_team"] == team) | (df["away_team"] == team)]
                if sub.empty:
                    continue
                mean_lambda = sub["λ"].mean()
                mean_mg13 = sub["MG1_3"].mean() * 100
                mean_mg14 = sub["MG1_4"].mean() * 100
                mean_mg15 = sub["MG1_5"].mean() * 100
                print(f"{team:<15} | n={len(sub):2d} | λ̄={mean_lambda:.2f} | MG1-3={mean_mg13:.1f}% | MG1-4={mean_mg14:.1f}% | MG1-5={mean_mg15:.1f}%")

        else:
            print("\n=== RISULTATI GLOBALI BACKTEST (range 1..k) ===")
            home_df = df[df["fav_side"] == "home"]
            away_df = df[df["fav_side"] == "away"]
            for lab, y, p in [("MG 1-3", "y1_3", "p1_3"), ("MG 1-4", "y1_4", "p1_4"), ("MG 1-5", "y1_5", "p1_5")]:
                for title, sub in [("Casa favorita", home_df), ("Ospite favorita", away_df)]:
                    bs = brier(sub[y], sub[p])
                    acc = float(((sub[p] >= 0.5) == (sub[y] == 1)).mean())
                    print(f"{title:<18} | {lab:<6} | n={len(sub):4d} | Brier={bs:.4f} | Acc={acc:.3f} | Media={sub[p].mean()*100:.1f}%")

            # --- STATISTICHE PER SQUADRA ---
            print("\n=== 📊 STATISTICHE PER SQUADRA TOP (STORICO) ===")
   
            for team in TOP_TEAMS:
                sub = df[(df["home_team"] == team) | (df["away_team"] == team)]
                if sub.empty:
                    continue
                print(f"\n--- {team} ---")
                for title, ssub in [("Casa favorita", sub[sub["fav_side"] == "home"]),
                                    ("Ospite favorita", sub[sub["fav_side"] == "away"])]:
                    for name, y, p in [("MG 1-3", "y1_3", "p1_3"),
                                       ("MG 1-4", "y1_4", "p1_4"),
                                       ("MG 1-5", "y1_5", "p1_5")]:
                        if ssub.empty:
                            continue
                        bs = brier(ssub[y], ssub[p])
                        acc = float(((ssub[p] >= 0.5) == (ssub[y] == 1)).mean())
                        print(f"{title:<18} | {name:<6} | n={len(ssub):4d} | Brier={bs:.4f} | Acc={acc:.3f} | Media={ssub[p].mean()*100:.1f}%")

    finally:
        db.close()
