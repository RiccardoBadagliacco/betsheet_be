"""
common/elo.py – Versione ROBUSTA
Funzioni PURE per calcolare Elo, riutilizzabili sia nel training che nel runtime.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import re

# ------------------------------
# PARAMETRI ELO
# ------------------------------
BASE_ELO = 1500.0
K = 18.0
H = 65.0
SEASON_DECAY = 0.65


# =======================================================
# NORMALIZZAZIONE STAGIONI
# =======================================================
def normalize_season_str(s: str) -> str | None:
    """
    Trasforma vari formati in 'YYYY/YYYY'.
    Supporta:
      - '2020/2021'
      - '2020_2021'
      - '2020-2021'
      - '2020/21'
      - '20/21'
      - '2021'
    """
    if not isinstance(s, str) or s.strip() == "":
        return None

    s = s.strip()

    # 1) Trova due anni in qualsiasi formato
    pattern_full = r"(\d{4})\D+(\d{4})"
    m = re.match(pattern_full, s)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        return f"{y1}/{y2}"

    # 2) Caso abbreviato: '20/21'
    pattern_short = r"(\d{2})\D+(\d{2})"
    m = re.match(pattern_short, s)
    if m:
        y1 = int(m.group(1))
        y2 = int(m.group(2))
        y1 = 2000 + y1
        y2 = 2000 + y2
        return f"{y1}/{y2}"

    # 3) Singolo anno → costruisco stagione Y/Y+1
    if re.match(r"^\d{4}$", s):
        y = int(s)
        return f"{y}/{y+1}"

    return None


def compute_season_order(seasons):
    """Ordina stagioni normalizzate."""
    norm = [normalize_season_str(s) for s in seasons]
    norm = [s for s in norm if s is not None]

    def key(s):
        return int(s.split("/")[0])

    return sorted(norm, key=key)


# =======================================================
# EXPECTED SCORE
# =======================================================
def expected_score(elo_home: float, elo_away: float) -> float:
    d = (elo_home + H) - elo_away
    return 1.0 / (1.0 + 10 ** (-d / 400))


# =======================================================
# COMPUTE FULL ELO HISTORY
# =======================================================
def compute_full_elo_history(df_step0: pd.DataFrame) -> pd.DataFrame:
    """
    df_step0 DEVE contenere:
      - match_id, date, season, home_team, away_team, home_ft, away_ft
    """
    required = [
        "match_id", "date", "season",
        "home_team", "away_team",
        "home_ft", "away_ft",
    ]
    for c in required:
        if c not in df_step0.columns:
            raise ValueError(f"Manca colonna richiesta per Elo: {c}")

    df = df_step0[required].copy()

    # Normalizzazione stagione
    df["season"] = df["season"].apply(normalize_season_str)
    df = df.dropna(subset=["season"])

    # Parsing data robusto
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Escludo match senza data
    df = df.dropna(subset=["date"])

    df = df.sort_values(["season", "date", "match_id"])

    seasons = compute_season_order(df["season"].unique())

    last_elo: dict[str, float] = {}
    rows = []

    for season in seasons:
        df_season = df[df["season"] == season]
        if df_season.empty:
            continue

        teams = sorted(set(df_season["home_team"]).union(df_season["away_team"]))

        # Elo iniziale
        elo = {}
        for t in teams:
            prev = last_elo.get(t)
            elo[t] = BASE_ELO if prev is None else BASE_ELO + (prev - BASE_ELO) * SEASON_DECAY

        # Loop partite
        for _, r in df_season.iterrows():
            home = r["home_team"]
            away = r["away_team"]
            gh = r["home_ft"]
            ga = r["away_ft"]

            # salta match senza risultato
            if pd.isna(gh) or pd.isna(ga):
                continue

            elo_home_pre = float(elo.get(home, BASE_ELO))
            elo_away_pre = float(elo.get(away, BASE_ELO))

            exp_home = expected_score(elo_home_pre, elo_away_pre)
            exp_away = 1.0 - exp_home

            # risultato 1-X-2
            if gh > ga:
                s_home = 1.0
            elif gh < ga:
                s_home = 0.0
            else:
                s_home = 0.5
            s_away = 1.0 - s_home

            elo_home_post = elo_home_pre + K * (s_home - exp_home)
            elo_away_post = elo_away_pre + K * (s_away - exp_away)

            elo[home] = elo_home_post
            elo[away] = elo_away_post

            rows.append({
                "match_id": r["match_id"],
                "date": r["date"],
                "season": season,
                "home_team": home,
                "away_team": away,
                "home_ft": gh,
                "away_ft": ga,
                "elo_home_pre": elo_home_pre,
                "elo_away_pre": elo_away_pre,
                "exp_home": exp_home,
                "exp_away": exp_away,
                "elo_home_post": elo_home_post,
                "elo_away_post": elo_away_post,
                "elo_diff": elo_home_pre - elo_away_pre,
            })

        # salva ELO finale della stagione
        for t, v in elo.items():
            last_elo[t] = v

    return pd.DataFrame(rows)


# =======================================================
# RUNTIME STATE
# =======================================================
def build_runtime_state_from_history(df_elo_history: pd.DataFrame) -> pd.DataFrame:
    required = ["match_id", "date", "home_team", "away_team", "elo_home_post", "elo_away_post"]
    for c in required:
        if c not in df_elo_history.columns:
            raise ValueError(f"Manca colonna Elo history: {c}")

    df = df_elo_history.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    home = df[["match_id", "date", "home_team", "elo_home_post"]].rename(
        columns={"home_team": "team", "elo_home_post": "elo"}
    )
    away = df[["match_id", "date", "away_team", "elo_away_post"]].rename(
        columns={"away_team": "team", "elo_away_post": "elo"}
    )

    all_rows = pd.concat([home, away], ignore_index=True)
    all_rows = all_rows.sort_values(["team", "date"])

    latest = all_rows.groupby("team").tail(1).reset_index(drop=True)
    latest = latest.rename(columns={"date": "last_match_date", "match_id": "last_match_id"})

    return latest[["team", "elo", "last_match_date", "last_match_id"]]