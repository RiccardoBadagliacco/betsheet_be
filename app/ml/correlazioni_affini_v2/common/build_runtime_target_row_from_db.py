import pandas as pd
import numpy as np

from app.ml.correlazioni_affini_v2.common.picchetto import apply_picchetto_tech_fix
from app.ml.correlazioni_affini_v2.common.elo import BASE_ELO
from app.ml.correlazioni_affini_v2.common.picchetto import DIFF_COLS_FIX


# ============================================================
# 1) FORM RUNTIME (SINGOLA PARTITA)
# ============================================================

def compute_form_lastN_runtime(df_history, team, match_date, last_n=6):
    """
    Replica compute_form_lastN MA per runtime (singola squadra).
    Usa df_history (step1c) e considera SOLO match prima di match_date.
    """
    df_team = df_history[
        ((df_history["home_team"] == team) | (df_history["away_team"] == team))
        & (df_history["date"] < match_date)
    ].sort_values("date", ascending=False).head(last_n)

    if df_team.empty:
        return dict(pts_avg=0, gf_avg=0, ga_avg=0)

    pts = []
    gf = []
    ga = []

    for _, r in df_team.iterrows():
        if r["home_team"] == team:
            gf.append(r["home_ft"])
            ga.append(r["away_ft"])
            pts.append(3 if r["home_ft"] > r["away_ft"] else 1 if r["home_ft"] == r["away_ft"] else 0)
        else:
            gf.append(r["away_ft"])
            ga.append(r["home_ft"])
            pts.append(3 if r["away_ft"] > r["home_ft"] else 1 if r["away_ft"] == r["home_ft"] else 0)

    return dict(
        pts_avg=float(np.mean(pts)),
        gf_avg=float(np.mean(gf)),
        ga_avg=float(np.mean(ga)),
    )


# ============================================================
# 2) ELO RUNTIME
# ============================================================

def get_last_elo(df_history, team):
    """
    Recupera l'ultimo elo_post (home o away) della squadra.
    """
    df_t = df_history[
        (df_history["home_team"] == team) | (df_history["away_team"] == team)
    ].sort_values("date", ascending=False)

    if df_t.empty:
        return BASE_ELO

    r = df_t.iloc[0]
    return (
        r["elo_home_post"]
        if r["home_team"] == team
        else r["elo_away_post"]
    )


# ============================================================
# 3) BUILD TARGET ROW (MATCH)
# ============================================================

def build_runtime_target_row_from_db(match, df_history, picchetto_stats):
    """
    Costruisce la riga runtime per un MATCH STORICO DA DB
    compatibile con PICCHETTO + CLUSTER + SOFT ENGINE.
    """

    home = match.home_team.name
    away = match.away_team.name
    match_date = pd.to_datetime(match.match_date)

    # -----------------------------
    # Quote bookmaker
    # -----------------------------
    odds1 = match.avg_home_odds
    oddsX = match.avg_draw_odds
    odds2 = match.avg_away_odds

    oddsO25 = match.avg_over_25_odds
    oddsU25 = match.avg_under_25_odds

    # -----------------------------
    # Forma
    # -----------------------------
    form_home = compute_form_lastN_runtime(df_history, home, match_date)
    form_away = compute_form_lastN_runtime(df_history, away, match_date)

    # -----------------------------
    # Elo
    # -----------------------------
    elo_home_pre = get_last_elo(df_history, home)
    elo_away_pre = get_last_elo(df_history, away)

    # -----------------------------
    # Costruisco DF base
    # -----------------------------
    row = {
        "match_id": str(match.id),
        "date": match_date,
        "home_team": home,
        "away_team": away,

        # bookmaker
        "bk_p1": 1/odds1 if odds1 else np.nan,
        "bk_px": 1/oddsX if oddsX else np.nan,
        "bk_p2": 1/odds2 if odds2 else np.nan,
        "bk_pO25": 1/oddsO25 if oddsO25 else np.nan,
        "bk_pU25": 1/oddsU25 if oddsU25 else np.nan,

        # elo
        "elo_home_pre": elo_home_pre,
        "elo_away_pre": elo_away_pre,
        "elo_diff_raw": elo_home_pre - elo_away_pre,

        # forma (già media ultimi N)
        "home_form_pts_avg_lastN": form_home["pts_avg"],
        "home_form_gf_avg_lastN": form_home["gf_avg"],
        "home_form_ga_avg_lastN": form_home["ga_avg"],

        "away_form_pts_avg_lastN": form_away["pts_avg"],
        "away_form_gf_avg_lastN": form_away["gf_avg"],
        "away_form_ga_avg_lastN": form_away["ga_avg"],
    }

    df_tmp = pd.DataFrame([row])

    # -----------------------------
    # PICCHETTO TECNICO
    # -----------------------------
    df_pic = apply_picchetto_tech_fix(df_tmp, stats=picchetto_stats)

    # -----------------------------
    # OUTPUT: una sola riga
    # -----------------------------
    return df_pic.iloc[0]