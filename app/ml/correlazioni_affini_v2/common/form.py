"""
common/form.py — VERSIONE ROBUSTA v3
Forma recente su rolling degli ultimi N match reali (no leakage)
Basata su date reali, non su season.
Disaggregazione home/away.
"""

from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np

DEFAULT_LAST_N = 6


def compute_form_lastN(df_step0: pd.DataFrame, last_n: int = DEFAULT_LAST_N) -> pd.DataFrame:
    """
    Calcola la forma recente basata sugli ultimi N match REALI (no leakage).
    Logica:
        - ordina per data
        - per ogni squadra costruisce uno storico completo
        - calcola per ogni match la forma pre-match della squadra di casa e trasferta

    OUTPUT per ogni match_id:
        - home_form_matches_lastN
        - home_form_pts_avg_lastN
        - home_form_gf_avg_lastN
        - home_form_ga_avg_lastN
        - home_form_win_rate_lastN
        - away_form_matches_lastN
        - away_form_pts_avg_lastN
        - away_form_gf_avg_lastN
        - away_form_ga_avg_lastN
        - away_form_win_rate_lastN
    """

    required = [
        "match_id", "date",
        "home_team", "away_team",
        "home_ft", "away_ft",
    ]
    for c in required:
        if c not in df_step0.columns:
            raise ValueError(f"Manca colonna richiesta: {c}")

    # Parse date
    df = df_step0[required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    # Output rows
    rows = []

    # storico per squadra: lista di dict {"gf":..., "ga":..., "pts":..., "is_win":...}
    history: Dict[str, list[Dict[str, Any]]] = {}

    for _, r in df.iterrows():
        mid = r["match_id"]
        home = r["home_team"]
        away = r["away_team"]
        gh = r["home_ft"]
        ga = r["away_ft"]

        # ==============================
        # Forma HOME pre-match
        # ==============================
        prev_home = history.get(home, [])
        last_home = prev_home[-last_n:]
        m_home = len(last_home)

        if m_home > 0:
            home_pts_avg = float(np.mean([x["pts"] for x in last_home]))
            home_gf_avg = float(np.mean([x["gf"] for x in last_home]))
            home_ga_avg = float(np.mean([x["ga"] for x in last_home]))
            home_win_rate = float(np.mean([x["is_win"] for x in last_home]))
        else:
            home_pts_avg = home_gf_avg = home_ga_avg = home_win_rate = 0.0

        # ==============================
        # Forma AWAY pre-match
        # ==============================
        prev_away = history.get(away, [])
        last_away = prev_away[-last_n:]
        m_away = len(last_away)

        if m_away > 0:
            away_pts_avg = float(np.mean([x["pts"] for x in last_away]))
            away_gf_avg = float(np.mean([x["gf"] for x in last_away]))
            away_ga_avg = float(np.mean([x["ga"] for x in last_away]))
            away_win_rate = float(np.mean([x["is_win"] for x in last_away]))
        else:
            away_pts_avg = away_gf_avg = away_ga_avg = away_win_rate = 0.0

        # ==============================
        # Riga di output PRE-match
        # ==============================
        rows.append({
            "match_id": mid,

            # numero di match disponibili negli ultimi N
            "home_form_matches_lastN": float(m_home),
            "away_form_matches_lastN": float(m_away),

            # medie HOME
            "home_form_pts_avg_lastN": home_pts_avg,
            "home_form_gf_avg_lastN": home_gf_avg,
            "home_form_ga_avg_lastN": home_ga_avg,
            "home_form_win_rate_lastN": home_win_rate,

            # medie AWAY
            "away_form_pts_avg_lastN": away_pts_avg,
            "away_form_gf_avg_lastN": away_gf_avg,
            "away_form_ga_avg_lastN": away_ga_avg,
            "away_form_win_rate_lastN": away_win_rate,
        })

        # ==============================
        # Aggiornamento storico POST-match
        # ==============================
        if pd.isna(gh) or pd.isna(ga):
            # fixture futura: non aggiorniamo history
            continue

        # update HOME history
        if gh > ga:
            pts_home = 3
            win_home = 1
        elif gh < ga:
            pts_home = 0
            win_home = 0
        else:
            pts_home = 1
            win_home = 0

        history.setdefault(home, []).append({
            "pts": pts_home,
            "gf": int(gh),
            "ga": int(ga),
            "is_win": win_home,
        })

        # update AWAY history
        if ga > gh:
            pts_away = 3
            win_away = 1
        elif ga < gh:
            pts_away = 0
            win_away = 0
        else:
            pts_away = 1
            win_away = 0

        history.setdefault(away, []).append({
            "pts": pts_away,
            "gf": int(ga),
            "ga": int(gh),
            "is_win": win_away,
        })

    return pd.DataFrame(rows).sort_values("match_id")


def compute_fixture_form(df_form: pd.DataFrame, home: str, away: str, date_fixture):
    """
    Calcola la forma (pts, gf, ga, win_rate) per una FIXTURE
    usando SOLO match con data < data_fixture.
    """
    df_form = df_form.copy()
    date_fixture = pd.to_datetime(date_fixture)

    # HOME
    f_home = df_form[
        ((df_form["home_form_matches_lastN"].notna()) | (df_form["away_form_matches_lastN"].notna()))
        &
        ((df_form["match_id"].notna()))  # safe
    ]

    # Trova tutte le righe di form dove home o away == team_home
    hist_home = f_home[
        (f_home["home_team"] == home) & (f_home["date"] < date_fixture)
    ].sort_values("date")

    if hist_home.empty:
        home_pts = home_gf = home_ga = home_win = 0.0
    else:
        last = hist_home.iloc[-1]
        home_pts = last["home_form_pts_avg_lastN"]
        home_gf  = last["home_form_gf_avg_lastN"]
        home_ga  = last["home_form_ga_avg_lastN"]
        home_win = last["home_form_win_rate_lastN"]

    # AWAY
    hist_away = f_home[
        (f_home["away_team"] == away) & (f_home["date"] < date_fixture)
    ].sort_values("date")

    if hist_away.empty:
        away_pts = away_gf = away_ga = away_win = 0.0
    else:
        last = hist_away.iloc[-1]
        away_pts = last["away_form_pts_avg_lastN"]
        away_gf  = last["away_form_gf_avg_lastN"]
        away_ga  = last["away_form_ga_avg_lastN"]
        away_win = last["away_form_win_rate_lastN"]

    return {
        "home_form_pts_avg_lastN": home_pts,
        "home_form_gf_avg_lastN": home_gf,
        "home_form_ga_avg_lastN": home_ga,
        "home_form_win_rate_lastN": home_win,

        "away_form_pts_avg_lastN": away_pts,
        "away_form_gf_avg_lastN": away_gf,
        "away_form_ga_avg_lastN": away_ga,
        "away_form_win_rate_lastN": away_win,
    }