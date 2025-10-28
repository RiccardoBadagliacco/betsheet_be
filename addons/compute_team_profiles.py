#!/usr/bin/env python3
"""
Modulo per il calcolo dei profili squadra su rolling window dalle partite in SQLite.
"""
import sqlite3
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict
from collections import defaultdict, deque

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
    init_team_elo(teams)
    build_team_profiles(df)
    
    
     # 🧭 Aggiorna ELO cronologicamente
    for _, row in df.iterrows():
        update_elo(row["home_team"], row["away_team"], row["home_goals_ft"], row["away_goals_ft"])
    
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
        
        # Over rate: % partite con più di 2.5 gol totali
        over_rate = ((team_matches["home_goals_ft"] + team_matches["away_goals_ft"]) > 2.5).mean()

        profiles[team] = {
            "gf": gf,
            "ga": ga,
            "over_rate": over_rate,
            "style": style,
            "elo": TEAM_ELO.get(team, 1500)
        }

    return profiles


TEAM_ELO = {}
TEAM_PROFILE = {}
def init_team_elo(teams, base_elo=1500):
    for t in teams:
        TEAM_ELO[t] = base_elo

def update_elo(home_team, away_team, home_goals, away_goals, k=20):
    eh = TEAM_ELO[home_team]
    ea = TEAM_ELO[away_team]
    expected_home = 1 / (1 + 10 ** ((ea - eh) / 400))
    actual_home = 1 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0
    TEAM_ELO[home_team] = eh + k * (actual_home - expected_home)
    TEAM_ELO[away_team] = ea + k * ((1 - actual_home) - (1 - expected_home))
    


def build_team_profiles(df):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    for team in teams:
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
        over_rate = ((matches['FTHG'] + matches['FTAG']) > 2.5).mean()
        style = 'attacking' if over_rate > 0.65 else 'defensive' if over_rate < 0.4 else 'neutral'
        TEAM_PROFILE[team] = {
            'over_rate': over_rate,
            'style': style
        }

def annotate_pre_match_elo(
    df: pd.DataFrame,
    base_elo: float = 1500.0,
    k: float = 20.0,
    window_size: int = 30,
    verbose: bool = False,
    show_history: bool = False  # 👈 nuovo flag per stampare lo storico
) -> pd.DataFrame:
    """
    Calcola l'ELO pre-partita per ogni match, considerando solo
    le ultime `window_size` partite per squadra.
    Se `show_history=True`, stampa le ultime partite usate per il calcolo.
    """
    required_cols = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"annotate_pre_match_elo: colonne mancanti: {missing}")

    dff = df.sort_values("Date").reset_index(drop=False)
    orig_index_col = "index"

    team_elo = defaultdict(lambda: base_elo)
    team_history = defaultdict(lambda: deque(maxlen=window_size))  # 👈 storico per squadra

    elo_home_pre, elo_away_pre = [], []

    if verbose:
        print(f"📊 Calcolo ELO (ultime {window_size} partite per squadra)")
        print(f"Totale match nel dataset: {len(dff)}\n")

    for i, row in dff.iterrows():
        
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        
        
        fthg = row["FTHG"]
        ftag = row["FTAG"]

        # 👇 Gestione match futuri (senza risultato)
        if fthg is None or ftag is None or pd.isna(fthg) or pd.isna(ftag):
            # assegna solo i pre-match ELO e continua
            eh, ea = team_elo[home], team_elo[away]
            elo_home_pre.append(eh)
            elo_away_pre.append(ea)
            continue
    
        hg, ag = float(row["FTHG"]), float(row["FTAG"])
        eh, ea = team_elo[home], team_elo[away]

        # Salva pre-match
        elo_home_pre.append(eh)
        elo_away_pre.append(ea)

        # Risultato effettivo
        if hg > ag:
            actual_home = 1.0
        elif hg < ag:
            actual_home = 0.0
        else:
            actual_home = 0.5

        expected_home = 1.0 / (1.0 + 10 ** ((ea - eh) / 400))
        actual_away = 1.0 - actual_home

        new_eh = eh + k * (actual_home - expected_home)
        new_ea = ea + k * (actual_away - (1.0 - expected_home))

        # Aggiorna storico ELO
        team_history[home].append(new_eh)
        team_history[away].append(new_ea)

        # Media delle ultime N partite
        team_elo[home] = sum(team_history[home]) / len(team_history[home])
        team_elo[away] = sum(team_history[away]) / len(team_history[away])

        # 🧾 Stampa debug
        if verbose and i < 10000:  # non esagerare
            print(f"[{i+1}] 📅 {row['Date']} {home} ({eh:.1f}) vs {away} ({ea:.1f}) → {int(hg)}-{int(ag)}")
            print(f"   🧮 expected_home={expected_home:.3f}, actual_home={actual_home:.1f}")
            print(f"   🆕 ELO post: {home}={team_elo[home]:.1f}, {away}={team_elo[away]:.1f}")
            
            # 👇 stampa le ultime partite usate nel calcolo
            if show_history:
                print(f"   📜 Ultime {len(team_history[home])} partite {home}: {list(map(lambda x: round(x,1), team_history[home]))}")
                print(f"   📜 Ultime {len(team_history[away])} partite {away}: {list(map(lambda x: round(x,1), team_history[away]))}")
            print("")

    dff["elo_home_pre"] = elo_home_pre
    dff["elo_away_pre"] = elo_away_pre

    out = df.copy()
    out.loc[dff[orig_index_col], "elo_home_pre"] = dff["elo_home_pre"].values
    out.loc[dff[orig_index_col], "elo_away_pre"] = dff["elo_away_pre"].values
    print(f"✅ Annotated ELO pre-match for {len(out)} matches")

    return out

def standardize_for_elo(df):
    """Rinomina le colonne dal formato originale al formato usato per ELO."""
    return df.rename(columns={
        "Date": "match_date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals_ft",
        "FTAG": "away_goals_ft"
    })

def restore_original_names(df):
    """Ripristina i nomi originali dopo il calcolo ELO."""
    return df.rename(columns={
        "match_date": "Date",
        "home_team": "HomeTeam",
        "away_team": "AwayTeam",
        "home_goals_ft": "FTHG",
        "away_goals_ft": "FTAG"
    })
    
def standardize_dataset_columns(df):
    """Rinomina le colonne al formato standard usato in tutta la pipeline."""
    return df.rename(columns={
        "Date": "match_date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals_ft",
        "FTAG": "away_goals_ft"
    })

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
