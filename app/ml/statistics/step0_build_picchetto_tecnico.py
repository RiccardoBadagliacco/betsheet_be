#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calcolo Picchetto Tecnico 1X2 per tutti i match del DB

- Usa il DB football (get_football_db)
- Joina:
    Match → Season → League → Team (home/away)
- Ordina cronologicamente per match_date
- Per ogni match:
    * Ricostruisce 4 righe statistiche (alla Stats4Bets):
        1) Generale stagione (tutte le partite giocate dalla squadra)
        2) Ultime 5 partite (totali)
        3) Casa/Trasferta stagione
        4) Ultime 5 in casa / trasferta
    * Calcola probabilità 1X2 per ogni riga
    * Fa la media delle 4 righe → Picchetto Tecnico:
        pt_p1, pt_px, pt_p2
    * Ricava quote tecniche:
        pt_q1 = 1/pt_p1, etc.

Output parquet:
    match_id, season, date, league,
    home_team_id, away_team_id, home_team, away_team,
    pt_p1, pt_px, pt_p2,
    pt_q1, pt_qx, pt_q2,
    gol_home, gol_away
"""

from __future__ import annotations

import sys
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session, aliased

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback se tqdm non installato
    def tqdm(x, total=None):
        return x

# Aggiungo la root del progetto al PYTHONPATH per l'esecuzione standalone
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.db.database_football import get_football_db
from app.db.models_football import Match, Season, League, Team


# -------------------------------------------------------------------------
# PATH OUTPUT
# -------------------------------------------------------------------------

# Salvo il parquet nella cartella "data" affiancata a questo script
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_OUT = DATA_DIR / "picchetto_tecnico_1x2.parquet"


# -------------------------------------------------------------------------
# STRUTTURE STORICO
# -------------------------------------------------------------------------

def build_stats_object() -> Dict[str, list | deque]:
    """
    Struttura per memorizzare lo storico di una squadra (per stagione, lato home/away).
    """
    return {
        "results": [],                 # tutte le partite (W/D/L)
        "results_last5": deque(maxlen=5),

        "results_home": [],            # solo quando gioca in casa
        "results_away": [],            # solo quando gioca in trasferta

        "results_last5_home": deque(maxlen=5),
        "results_last5_away": deque(maxlen=5),
    }


def convert_result(gf: int, ga: int) -> str:
    """
    Converte (gol fatti, subiti) in risultato W/D/L.
    """
    if gf > ga:
        return "W"
    if gf < ga:
        return "L"
    return "D"


def stats_summary(results: list[str]) -> Dict[str, int]:
    """
    Riepilogo statistico per una lista di risultati W/D/L.
    """
    if len(results) == 0:
        return {"played": 0, "wins": 0, "draws": 0, "losses": 0}

    arr = np.array(results, dtype=object)
    return {
        "played": len(arr),
        "wins": int(np.sum(arr == "W")),
        "draws": int(np.sum(arr == "D")),
        "losses": int(np.sum(arr == "L")),
    }


# -------------------------------------------------------------------------
# PICCHETTO TECNICO (singolo match)
# -------------------------------------------------------------------------

def compute_row_probs(stats_H: Dict[str, int], stats_A: Dict[str, int]) -> Tuple[float, float, float]:
    """
    Calcola le probabilità grezze di 1X2 da una singola riga statistica
    (versione Stats4Bets semplificata):

        P1 = (vittorie_home + sconfitte_away) / (match_home + match_away)
        P2 = (vittorie_away + sconfitte_home) / (match_home + match_away)
        PX = 1 - (P1 + P2)
    """
    partite_H = stats_H["played"]
    partite_A = stats_A["played"]

    tot = partite_H + partite_A
    if tot == 0:
        return (np.nan, np.nan, np.nan)

    P1 = (stats_H["wins"] + stats_A["losses"]) / tot
    P2 = (stats_A["wins"] + stats_H["losses"]) / tot
    PX = 1.0 - (P1 + P2)

    return (float(P1), float(PX), float(P2))


def compute_picchetto_for_match(
    row: pd.Series,
    hist_H: Dict[str, list | deque],
    hist_A: Dict[str, list | deque],
) -> Tuple[float, float, float]:
    """
    row = record del match corrente (home, away, goals, ecc.)
    hist_H, hist_A = storico della squadra di casa / away fino a *prima* di questo match.
    """

    # Riga 1: generale stagione (tutti i match)
    H_gen = stats_summary(hist_H["results"])
    A_gen = stats_summary(hist_A["results"])

    # Riga 2: ultime 5 partite complessive
    H_last5 = stats_summary(list(hist_H["results_last5"]))
    A_last5 = stats_summary(list(hist_A["results_last5"]))

    # Riga 3: casa/trasferta stagionale
    H_home = stats_summary(hist_H["results_home"])
    A_away = stats_summary(hist_A["results_away"])

    # Riga 4: ultime 5 home/away
    H_last5_home = stats_summary(list(hist_H["results_last5_home"]))
    A_last5_away = stats_summary(list(hist_A["results_last5_away"]))

    rows_probs = []

    # Applico la formula riga per riga
    for rH, rA in [
        (H_gen, H_gen),            # generale vs generale
        (H_last5, A_last5),        # ultime 5 vs ultime 5
        (H_home, A_away),          # casa vs trasferta
        (H_last5_home, A_last5_away),
    ]:
        P1, PX, P2 = compute_row_probs(rH, rA)
        rows_probs.append((P1, PX, P2))

    arr = np.array(rows_probs, dtype=float)

    # Media lungo le 4 righe (ignora NaN)
    PT_1, PT_X, PT_2 = np.nanmean(arr, axis=0)

    return float(PT_1), float(PT_X), float(PT_2)


# -------------------------------------------------------------------------
# CARICAMENTO MATCH DAL DB
# -------------------------------------------------------------------------

def load_matches_from_db(session: Session) -> pd.DataFrame:
    """
    Legge TUTTE le partite storiche dal DB football, con join:
    Match → Season → League → Team (home, away).

    Ritorna un DataFrame ordinato per data.
    """
    print("📥 Lettura MATCH dal database…")

    HomeTeam = aliased(Team)
    AwayTeam = aliased(Team)

    q = (
        session.query(
            Match.id.label("match_id"),
            Match.match_date.label("match_date"),
            Match.home_goals_ft.label("home_goals_ft"),
            Match.away_goals_ft.label("away_goals_ft"),
            Season.code.label("season"),          # es. "2324"
            League.code.label("league"),          # es. "I1", "E0"
            HomeTeam.id.label("home_team_id"),
            HomeTeam.name.label("home_team"),
            AwayTeam.id.label("away_team_id"),
            AwayTeam.name.label("away_team"),
        )
        .join(Season, Match.season_id == Season.id)
        .join(League, Season.league_id == League.id)
        .join(HomeTeam, Match.home_team_id == HomeTeam.id)
        .join(AwayTeam, Match.away_team_id == AwayTeam.id)
        .filter(Match.home_goals_ft.isnot(None))
        .filter(Match.away_goals_ft.isnot(None))
    )

    rows = q.order_by(Match.match_date.asc(), Match.id.asc()).all()
    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("Nessun match storico trovato nel DB.")

    # match_date potrebbe essere SafeDate → porto a datetime/Date coerente
    df["match_date"] = pd.to_datetime(df["match_date"])

    print(f"➡️  Trovati {len(df)} match storici con risultato FT.")
    return df


# -------------------------------------------------------------------------
# PIPELINE COMPLETA PICCHETTO TECNICO
# -------------------------------------------------------------------------

def build_picchetto_dataset(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Esegue il calcolo del Picchetto Tecnico per tutti i match nel DataFrame.

    df_matches deve avere le colonne:
      - match_id
      - match_date
      - season
      - league
      - home_team_id, home_team
      - away_team_id, away_team
      - home_goals_ft, away_goals_ft
    """

    # Ordino in modo garantito
    df = df_matches.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    # Storici per (team_id, season)
    history: Dict[Tuple[str, str], Dict[str, list | deque]] = defaultdict(build_stats_object)

    out_rows = []

    print("🧮 Calcolo Picchetto Tecnico per ogni match…")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        season = str(row["season"])
        h_id = str(row["home_team_id"])
        a_id = str(row["away_team_id"])

        key_H = (h_id, season)
        key_A = (a_id, season)

        hist_H = history[key_H]
        hist_A = history[key_A]

        # Calcolo PT PRIMA di aggiornare lo storico
        pt_p1, pt_px, pt_p2 = compute_picchetto_for_match(row, hist_H, hist_A)

        # Quote tecniche
        pt_q1 = 1.0 / pt_p1 if pt_p1 and pt_p1 > 0 else np.nan
        pt_qx = 1.0 / pt_px if pt_px and pt_px > 0 else np.nan
        pt_q2 = 1.0 / pt_p2 if pt_p2 and pt_p2 > 0 else np.nan

        out_rows.append(
            {
                # Converto in stringa per compatibilità parquet/pyarrow
                "match_id": str(row["match_id"]),
                "season": season,
                "date": row["match_date"],
                "league": row["league"],
                "home_team_id": h_id,
                "away_team_id": a_id,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "gol_home": int(row["home_goals_ft"]),
                "gol_away": int(row["away_goals_ft"]),
                "pt_p1": pt_p1,
                "pt_px": pt_px,
                "pt_p2": pt_p2,
                "pt_q1": pt_q1,
                "pt_qx": pt_qx,
                "pt_q2": pt_q2,
            }
        )

        # ---------------------------------------------------
        # AGGIORNO LO STORICO DOPO IL MATCH
        # ---------------------------------------------------
        hg = int(row["home_goals_ft"])
        ag = int(row["away_goals_ft"])

        res_H = convert_result(hg, ag)
        res_A = convert_result(ag, hg)

        # generale
        hist_H["results"].append(res_H)
        hist_A["results"].append(res_A)

        hist_H["results_last5"].append(res_H)
        hist_A["results_last5"].append(res_A)

        # home/away
        hist_H["results_home"].append(res_H)
        hist_A["results_away"].append(res_A)

        hist_H["results_last5_home"].append(res_H)
        hist_A["results_last5_away"].append(res_A)

    out_df = pd.DataFrame(out_rows)
    return out_df


def run():
    """
    Entrypoint principale:
    - apre una sessione DB
    - carica i match
    - calcola il picchetto
    - salva in parquet
    """
    session: Session = next(get_football_db())

    try:
        df_matches = load_matches_from_db(session)
        pic_df = build_picchetto_dataset(df_matches)

        pic_df.to_parquet(PARQUET_OUT, index=False)
        print(f"💾 Picchetto tecnico salvato in: {PARQUET_OUT}")
        print(f"📊 Righe salvate: {len(pic_df)}")

    finally:
        session.close()


if __name__ == "__main__":
    run()
