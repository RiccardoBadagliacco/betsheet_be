#!/usr/bin/env python3
"""
Modulo per il calcolo dei profili squadra su rolling window dalle partite in SQLite.
"""
import sqlite3
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict

# Configura logging opzionale
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("team_profiles")

def compute_team_profiles(rolling_n: int = 15, season_filter: Optional[str] = None, use_weighted: bool = True) -> Dict[str, dict]:
    """
    Calcola i profili delle squadre su rolling window dalle ultime N partite.
    Args:
        rolling_n: numero di partite recenti da considerare
        season_filter: filtra per stagione (es. "2025/26")
        use_weighted: se True usa media pesata esponenzialmente
    Returns:
        dict: {team: {gf, ga, style}}
    """
    # Connessione al database
    db_path = "data/football_dataset.db"
    conn = sqlite3.connect(db_path)
    # Join per ottenere i nomi delle squadre e stagione
    query = """
        SELECT m.match_date, s.code as season, ht.name as home_team, at.name as away_team,
               m.home_goals_ft, m.away_goals_ft
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        JOIN seasons s ON m.season_id = s.id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Parsing date
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date", "home_team", "away_team"])  # Rimuovi righe incomplete
    if season_filter:
        df = df[df["season"] == season_filter]

    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    profiles = {}

    for team in teams:
        # Partite dove la squadra è coinvolta
        mask_home = df["home_team"] == team
        mask_away = df["away_team"] == team
        team_matches = df[mask_home | mask_away].copy()
        team_matches = team_matches.sort_values("match_date", ascending=False).head(rolling_n)

        if len(team_matches) < 5:
            profiles[team] = {"gf": 1.5, "ga": 1.5, "style": "balanced"}
            continue

        # Calcola goal fatti/subiti per ogni match
        def get_gf(row):
            return row["home_goals_ft"] if row["home_team"] == team else row["away_goals_ft"]
        def get_ga(row):
            return row["away_goals_ft"] if row["home_team"] == team else row["home_goals_ft"]
        team_matches["gf"] = team_matches.apply(get_gf, axis=1)
        team_matches["ga"] = team_matches.apply(get_ga, axis=1)

        if use_weighted:
            # Pesatura esponenziale: più peso alle partite recenti
            weights = np.exp(np.linspace(0, -2, len(team_matches)))
            weights = weights / weights.sum()
            gf = np.average(team_matches["gf"], weights=weights)
            ga = np.average(team_matches["ga"], weights=weights)
        else:
            gf = team_matches["gf"].mean()
            ga = team_matches["ga"].mean()

        # Classificazione stile
        if gf >= 2.0:
            style = "attacking"
        elif gf <= 1.2 and ga <= 1.0:
            style = "defensive"
        else:
            style = "balanced"

        profiles[team] = {"gf": round(gf, 2), "ga": round(ga, 2), "style": style}

    return profiles

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calcola i profili delle squadre da football_dataset.db")
    parser.add_argument("--rolling_n", type=int, default=15, help="Numero di partite recenti da considerare")
    parser.add_argument("--season", type=str, default=None, help="Filtra per stagione (es. 2025/26)")
    parser.add_argument("--no-weighted", action="store_true", help="Disabilita media pesata esponenziale")
    args = parser.parse_args()

    profiles = compute_team_profiles(
        rolling_n=args.rolling_n,
        season_filter=args.season,
        use_weighted=not args.no_weighted
    )
    print(f"\n{'Team':<25}  GF   GA   Style")
    print("-"*45)
    for team, prof in sorted(profiles.items()):
        print(f"{team:<25}  {prof['gf']:.2f}  {prof['ga']:.2f}  {prof['style']}")
