#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5 — BACKTEST FAVORITE_PROFILE_SIGNAL

Obiettivo:
- Per ogni match storico con soft_probs e dati wide:
    * costruire il profilo favorita
    * applicare rule_favorite_profile_signal
    * confrontare le giocate suggerite con il risultato reale

Output:
- Statistiche per scenario:
    * NEGATIVE_RISK_NOWIN_NOGOAL
    * STRONG_WIN_AND_GOAL
    * MULTIGOAL_1_3 / 1_4 / 1_5
    * STANDARD_PROFILE
    * nessun alert / nessuna favorita

- Statistiche per bet_tag:
    * numero di partite
    * win rate empirico

Uso:
    python step5_backtest_favorite_profile_signal.py --n 20000
"""

import sys
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
from tqdm import tqdm

# -----------------------
# PATH & IMPORT
# -----------------------
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR   = Path(__file__).resolve().parents[2]
DATA_DIR   = BASE_DIR / "data"

SOFT_FILE  = DATA_DIR / "step5_soft_history.parquet"
WIDE_FILE  = DATA_DIR / "step4a_affini_index_wide_v2.parquet"

from app.ml.correlazioni_affini_v2.common.betting_rules.rule_favorite_profile_signal import (
    rule_favorite_profile_signal,
)


# --------------------------------------------------------------------
# Helper per valutare le giocate suggerite
# --------------------------------------------------------------------

def _eval_single_bet(
    tag: str,
    side: str,
    home_ft: int,
    away_ft: int,
) -> Optional[bool]:
    """
    Traduce un bet_tag in una condizione booleana (vinta / persa).

    Ritorna:
        True  -> bet vinta
        False -> bet persa
        None  -> bet non valutata in questo backtest (es. Asian -0.25)
    """
    tg = home_ft + away_ft
    if side == "home":
        fav_goals = home_ft
        opp_goals = away_ft
    else:
        fav_goals = away_ft
        opp_goals = home_ft

    # Tag "gestionali" → non sono vere giocate
    if tag in {"BET_NO_BET_STRONG"}:
        return None

    # UNDER globali
    if tag == "BET_UNDER15":
        return tg <= 1
    if tag == "BET_UNDER25":
        return tg <= 2

    # UNDER favorita (gol squadra)
    if tag == "BET_UNDER_1_5_FAV":
        return fav_goals <= 1

    # Lay favorita segna = banca "favorita segna"
    if tag == "BET_LAY_FAV_SEGNA":
        return fav_goals == 0

    # Segno fisso
    if tag == "BET_1":
        return home_ft > away_ft
    if tag == "BET_2":
        return away_ft > home_ft

    # Doppie chance
    if tag == "BET_DC_OPPOSITE":
        # DC a favore dell'altra squadra
        # home fav -> X2  (vinta se home non vince)
        # away fav -> 1X  (vinta se away non vince)
        if side == "home":
            return home_ft <= away_ft  # X o 2
        else:
            return away_ft <= home_ft  # X o 1

    if tag == "BET_DC_FAV":
        # DC a favore favorita
        # home fav -> 1X, away fav -> X2
        if side == "home":
            return home_ft >= away_ft  # 1 o X
        else:
            return away_ft >= home_ft  # 2 o X

    # Gol favorita
    if tag in {"BET_FAV_SEGNA", "BET_FAV_OVER_0_5"}:
        return fav_goals >= 1

    # Multigol favorita
    if tag == "BET_FAV_1_3":
        return 1 <= fav_goals <= 3
    if tag == "BET_FAV_1_4":
        return 1 <= fav_goals <= 4
    if tag == "BET_FAV_1_5":
        return 1 <= fav_goals <= 5

    # Asian -0.25 favorita: per ora NON lo valutiamo (richiede gestione half-win)
    if tag == "BET_ASIAN_-0_25_FAV":
        return None

    # Tag sconosciuto → non lo valutiamo
    return None


def _extract_raw_bet_tags_from_alert(alert) -> List[str]:
    """
    Rende il backtest robusto ai cambi sul modello di BettingAlert.

    Supporta:
    - versione vecchia: alert.bets = ["BET_DC_FAV", ...]
    - versione nuova:  alert.bets = [{"tag_raw": "...", ...}, ...]
    - versione nuova con meta["bet_tags_raw"] = ["BET_DC_FAV", ...]
    """
    raw_tags: List[str] = []

    meta = getattr(alert, "meta", {}) or {}

    # 1) Se esiste meta["bet_tags_raw"], è la sorgente ufficiale
    bt = meta.get("bet_tags_raw")
    if isinstance(bt, list):
        for x in bt:
            if isinstance(x, str):
                raw_tags.append(x)
        if raw_tags:
            return raw_tags

    # 2) Altrimenti proviamo a usare alert.bets
    bets_field = getattr(alert, "bets", None) or []
    for b in bets_field:
        if isinstance(b, str):
            raw_tags.append(b)
        elif isinstance(b, dict):
            if "tag_raw" in b and isinstance(b["tag_raw"], str):
                raw_tags.append(b["tag_raw"])
            elif "code" in b and isinstance(b["code"], str):
                raw_tags.append(b["code"])

    # Deduplica mantenendo l'ordine
    seen = set()
    deduped: List[str] = []
    for t in raw_tags:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return deduped


# --------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------

def run_backtest(n_matches: Optional[int] = None) -> None:
    # ---------------------------
    # 1. Carica soft_history + wide
    # ---------------------------
    print(f"📥 Carico soft_history da: {SOFT_FILE}")
    df_soft = pd.read_parquet(SOFT_FILE)

    print(f"📥 Carico wide da:        {WIDE_FILE}")
    wide = pd.read_parquet(WIDE_FILE)

    # Seleziono dal wide solo le colonne che servono alla regola
    wide_cols_needed = [
        "match_id",
        "bk_p1", "bk_px", "bk_p2",
        "pic_p1", "pic_p2",
        "tightness_index",
        "lambda_total_form",
        "cluster_1x2",
        "cluster_ou25",
        "home_ft", "away_ft",
    ]
    missing = [c for c in wide_cols_needed if c not in wide.columns]
    if missing:
        raise RuntimeError(f"Mancano colonne nel wide: {missing}")

    wide_sel = wide[wide_cols_needed].copy()

    # Merge soft_history + wide su match_id
    df = df_soft.merge(wide_sel, on="match_id", how="inner", suffixes=("", "_w"))

    # Filtra solo status ok
    df = df[df["status"] == "ok"].copy()

    if n_matches is not None and n_matches < len(df):
        df = df.sample(n_matches, random_state=42).copy()
        print(f"✂️  Uso un sottoinsieme di {len(df)} partite per il backtest.")
    else:
        print(f"➡️  Backtest su tutte le partite disponibili: {len(df)}")

    # ---------------------------
    # 2. Loop sui match
    # ---------------------------
    scenario_stats: Dict[str, Dict[str, float]] = {}
    # scenario -> {"n": int, "fav_wins": int, "fav_scores": int}

    bet_stats: Dict[str, Dict[str, float]] = {}
    # tag -> {"n": int, "wins": int}

    no_favorite_count = 0
    no_alert_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Backtesting", unit="match"):
        # Prepara ctx per la regola
        soft_probs = {
            "p1":  float(row["soft_p1"]),
            "px":  float(row["soft_px"]),
            "p2":  float(row["soft_p2"]),
            "pO15": float(row["soft_pO15"]),
            "pU15": float(row["soft_pU15"]),
            "pO25": float(row["soft_pO25"]),
            "pU25": float(row["soft_pU25"]),
        }
        ctx: Dict[str, Any] = {"soft_probs": soft_probs}

        # Applica la regola
        alerts = rule_favorite_profile_signal(row, ctx)

        if not alerts:
            # o non c'è favorita, o profilo neutro
            # provo a vedere se c'è favorita a livello tecnico (bk_p1/bk_p2)
            bk_p1 = float(row["bk_p1"])
            bk_px = float(row["bk_px"])
            bk_p2 = float(row["bk_p2"])
            is_home_fav = (bk_p1 > max(bk_px, bk_p2))
            is_away_fav = (bk_p2 > max(bk_px, bk_p1))
            if is_home_fav or is_away_fav:
                no_alert_count += 1
            else:
                no_favorite_count += 1
            continue

        # La regola è progettata per restituire al massimo 1 alert
        alert = alerts[0]

        meta = alert.meta or {}
        side = meta.get("side")
        scenario = meta.get("scenario", "UNKNOWN")

        home_ft = int(row["home_ft"])
        away_ft = int(row["away_ft"])

        if side not in {"home", "away"}:
            # fallback: ricalcolo da bk_p1/bk_p2
            bk_p1 = float(row["bk_p1"])
            bk_px = float(row["bk_px"])
            bk_p2 = float(row["bk_p2"])
            is_home_fav = (bk_p1 > max(bk_px, bk_p2))
            side = "home" if is_home_fav else "away"

        if side == "home":
            fav_goals = home_ft
            opp_goals = away_ft
        else:
            fav_goals = away_ft
            opp_goals = home_ft

        fav_wins = fav_goals > opp_goals
        fav_scores = fav_goals > 0

        # ---------------------------
        # 2a. Aggiorna stats per scenario
        # ---------------------------
        sc = scenario_stats.setdefault(scenario, {
            "n": 0,
            "fav_wins": 0,
            "fav_scores": 0,
        })
        sc["n"] += 1
        sc["fav_wins"] += int(fav_wins)
        sc["fav_scores"] += int(fav_scores)

        # ---------------------------
        # 2b. Aggiorna stats per bet_tag (USANDO I TAG RAW)
        # ---------------------------
        raw_tags = _extract_raw_bet_tags_from_alert(alert)
        for tag in raw_tags:
            res = _eval_single_bet(tag, side, home_ft, away_ft)
            if res is None:
                continue
            bstat = bet_stats.setdefault(tag, {"n": 0, "wins": 0})
            bstat["n"] += 1
            bstat["wins"] += int(res)

    # ---------------------------
    # 3. Report finale
    # ---------------------------
    print("\n==============================================================")
    print("📊 RISULTATI BACKTEST FAVORITE_PROFILE_SIGNAL")
    print("==============================================================\n")

    print(f"Totale match analizzati:     {len(df)}")
    print(f"Match senza favorita:        {no_favorite_count}")
    print(f"Match con favorita ma senza alert: {no_alert_count}")
    print()

    # Scenario stats
    if scenario_stats:
        print("--------------------------------------------------------------")
        print("📌 Statistiche per SCENARIO")
        print("--------------------------------------------------------------")
        rows_scen = []
        for scen, s in scenario_stats.items():
            n = s["n"]
            fav_win_rate = s["fav_wins"] / n if n > 0 else math.nan
            fav_score_rate = s["fav_scores"] / n if n > 0 else math.nan
            rows_scen.append({
                "scenario": scen,
                "n": n,
                "fav_win_rate": fav_win_rate,
                "fav_score_rate": fav_score_rate,
            })

        df_scen = pd.DataFrame(rows_scen).sort_values("n", ascending=False)
        print(df_scen.to_string(index=False))
        print()

    # Bet stats
    if bet_stats:
        print("--------------------------------------------------------------")
        print("🎯 Statistiche per BET TAG")
        print("--------------------------------------------------------------")
        rows_bet = []
        for tag, s in bet_stats.items():
            n = s["n"]
            win_rate = s["wins"] / n if n > 0 else math.nan
            rows_bet.append({
                "bet_tag": tag,
                "n": n,
                "win_rate": win_rate,
            })

        df_bet = pd.DataFrame(rows_bet).sort_values("n", ascending=False)
        print(df_bet.to_string(index=False))
        print()

    print("🏁 BACKTEST COMPLETATO.")


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Numero massimo di partite da usare (sample). Se omesso usa tutte.",
    )
    args = parser.parse_args()

    run_backtest(n_matches=args.n)


if __name__ == "__main__":
    main()