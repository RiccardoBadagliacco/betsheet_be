import numpy as np

def shin_derake(odds, max_iter=100, tol=1e-6):
    """
    Shin (1993) deraking for 3-way odds.
    Returns: (probabilities, zeta)
    """
    inv = np.array([1/x for x in odds], float)
    s = inv.sum()
    z = 0.05  # initial guess for insider proportion

    for _ in range(max_iter):
        p = (1 - z) * inv / (1 - z * inv / s)
        diff = p.sum() - 1
        if abs(diff) < tol:
            break
        z -= diff * 0.5 * (1 - z)
        z = np.clip(z, 0.0, 0.25)

    p /= p.sum()
    return p.tolist(), z

import sqlite3, pandas as pd

def build_mu_priors(db_path="./data/football_dataset.db"):
    """
    Calcola i gol medi totali (home+away) per ogni lega e stagione.
    """
    q = """
    SELECT l.code AS league_code, s.name AS season,
           AVG(m.home_goals_ft + m.away_goals_ft) AS mu_total
    FROM matches m
    JOIN seasons s ON s.id = m.season_id
    JOIN leagues l ON l.id = s.league_id
    WHERE m.home_goals_ft IS NOT NULL
    GROUP BY l.code, s.name
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(q, conn)
    conn.close()
    
    res = {}
    for _, r in df.iterrows():
        league, season = r.league_code, r.season
        mu = float(r.mu_total)
        res[(league, season)] = mu
        # chiave alternativa
        res[(league, season.replace("_", "/"))] = mu
    return res

import sqlite3
import pandas as pd
import numpy as np

def compute_recent_form(db_path="./data/football_dataset.db", window=5, alpha=0.3):
    """
    Calcola la forma recente di ogni squadra (gol fatti/subiti medi nelle ultime N partite).
    Usa una media mobile esponenziale (EMA) per dare più peso alle partite recenti.
    """
    import sqlite3, pandas as pd
    conn = sqlite3.connect(db_path)
    q = f"""
    SELECT
        t.id AS team_id,
        s.name AS season,
        m.match_date,
        CASE WHEN m.home_team_id = t.id THEN m.home_goals_ft ELSE m.away_goals_ft END AS goals_for,
        CASE WHEN m.home_team_id = t.id THEN m.away_goals_ft ELSE m.home_goals_ft END AS goals_against
    FROM matches m
    JOIN seasons s ON s.id = m.season_id
    JOIN teams t ON t.id IN (m.home_team_id, m.away_team_id)
    WHERE m.home_goals_ft IS NOT NULL
    ORDER BY t.id, m.match_date ASC
    """
    df = pd.read_sql(q, conn)
    conn.close()

    # Calcola la forma con media mobile esponenziale (più peso alle ultime partite)
    df["gf_recent"] = (
        df.groupby("team_id")["goals_for"]
          .apply(lambda x: x.ewm(span=window, adjust=False).mean())
          .reset_index(level=0, drop=True)
    )
    df["ga_recent"] = (
        df.groupby("team_id")["goals_against"]
          .apply(lambda x: x.ewm(span=window, adjust=False).mean())
          .reset_index(level=0, drop=True)
    )

    # Forma "stagionale" di base per smussare (in caso di stagioni corte o partite poche)
    mean_form = (
        df.groupby(["team_id", "season"])[["goals_for", "goals_against"]]
          .mean()
          .rename(columns={"goals_for": "gf_mean", "goals_against": "ga_mean"})
          .reset_index()
    )

    # Prendi l’ultimo record per squadra e stagione (stato attuale)
    form = df.groupby(["team_id", "season"]).tail(1).reset_index(drop=True)

    # Unisci con la forma media stagionale
    form = form.merge(mean_form, on=["team_id", "season"], how="left")

    # Smooth tra forma recente (EMA) e media stagionale
    form["gf_recent"] = alpha * form["gf_recent"] + (1 - alpha) * form["gf_mean"]
    form["ga_recent"] = alpha * form["ga_recent"] + (1 - alpha) * form["ga_mean"]

    return form[["team_id", "season", "gf_recent", "ga_recent"]]


