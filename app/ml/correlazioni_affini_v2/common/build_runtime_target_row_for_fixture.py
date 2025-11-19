import pandas as pd
import numpy as np
from app.ml.correlazioni_affini_v2.common.form import compute_form_lastN
from app.ml.correlazioni_affini_v2.common.elo import build_runtime_state_from_history
from app.ml.correlazioni_affini_v2.common.picchetto import apply_picchetto_tech_fix

def build_runtime_target_row_for_fixture(
    fixture,
    df_history: pd.DataFrame,
    picchetto_stats: dict,
) -> pd.Series:
    """
    Costruisce una singola riga runtime per la fixture futura,
    con tutte le feature necessarie allo slim_index / clustering.
    """

    # ========== 1) Forma recente ==========
    df_form = compute_form_lastN(df_history)
    last_form = df_form[df_form["match_id"] == df_form["match_id"].max()]  # NON usata, solo schema

    # prendi forma home
    home_hist = df_form[df_form["home_team"] == fixture.home_team.name]
    away_hist = df_form[df_form["home_team"] == fixture.away_team.name]

    home_last = home_hist.tail(1).iloc[0] if not home_hist.empty else None
    away_last = away_hist.tail(1).iloc[0] if not away_hist.empty else None

    # ========== 2) Elo recente ==========
    df_elo_hist = df_history[
        ["match_id", "date", "home_team", "away_team", "home_ft", "away_ft"]
    ].copy()

    from .elo import compute_full_elo_history
    df_elo = compute_full_elo_history(df_history)

    runtime_elo = build_runtime_state_from_history(df_elo)
    elo_home = runtime_elo[runtime_elo["team"] == fixture.home_team.name]["elo"].iloc[0]
    elo_away = runtime_elo[runtime_elo["team"] == fixture.away_team.name]["elo"].iloc[0]

    elo_diff = elo_home - elo_away

    # ========== 3) Costruisco row base ==========
    base = dict(
        match_id=fixture.id,
        home_team=fixture.home_team.name,
        away_team=fixture.away_team.name,
        bk_p1 = fixture.odds.get("p1"),
        bk_px = fixture.odds.get("px"),
        bk_p2 = fixture.odds.get("p2"),
        bk_pO25 = fixture.odds.get("pO25"),
        bk_pU25 = fixture.odds.get("pU25"),
        home_form_pts_avg_lastN = home_last["home_form_pts_avg_lastN"] if home_last is not None else 0,
        home_form_gf_avg_lastN  = home_last["home_form_gf_avg_lastN"]  if home_last is not None else 0,
        home_form_ga_avg_lastN  = home_last["home_form_ga_avg_lastN"]  if home_last is not None else 0,
        away_form_pts_avg_lastN = away_last["home_form_pts_avg_lastN"] if away_last is not None else 0,
        away_form_gf_avg_lastN  = away_last["home_form_gf_avg_lastN"]  if away_last is not None else 0,
        away_form_ga_avg_lastN  = away_last["home_form_ga_avg_lastN"]  if away_last is not None else 0,
        elo_home_pre=elo_home,
        elo_away_pre=elo_away,
        elo_diff=elo_diff,
    )

    df = pd.DataFrame([base])

    # ========== 4) Applico picchetto tecnico ==========
    df = apply_picchetto_tech_fix(df, stats=picchetto_stats)

    # ========== 5) Calcolo delta prob ==========
    df["delta_p1"] = df["pic_p1"] - df["bk_p1"]
    df["delta_p2"] = df["pic_p2"] - df["bk_p2"]
    df["market_sharpness"] = (
        abs(df["delta_p1"]) + abs(df["delta_p2"]) + abs(df["pic_px"] - df["bk_px"])
    )

    # ========== 6) Lambda form ==========
    df["lambda_total_form"] = (
        df["home_form_gf_avg_lastN"] + df["away_form_gf_avg_lastN"]
    )

    # ========== 7) Lambda market ==========
    df["lambda_total_market_ou25"] = None
    if df["bk_pO25"].notna().iloc[0] and df["bk_pU25"].notna().iloc[0]:
        p = df["bk_pO25"].iloc[0]
        lam = np.log(1 / (1 - p))  # approx per line 2.5
        df["lambda_total_market_ou25"] = lam

    # calcolo delta O25/U25
    df["delta_O25"] = df["pic_pO25"] - df["bk_pO25"]
    df["delta_U25"] = df["pic_pU25"] - df["bk_pU25"]

    return df.iloc[0]