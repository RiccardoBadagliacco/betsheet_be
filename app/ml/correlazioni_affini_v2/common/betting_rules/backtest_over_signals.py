#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5 — BACKTEST OVER_UNDER_SIGNAL

Obiettivo:
- Valutare la nuova regola Over/Under (v2.0)
- Per ogni match:
    * costruire soft_probs
    * applicare rule_over_signal
    * confrontare i bet_tags con risultato reale

Output:
- Statistiche per SCENARIO:
    * STRONG_OVER
    * OVER_EDGE
    * STRONG_UNDER
    * UNDER_EDGE
    * BALANCED_OVER_EDGE
    * BALANCED_UNDER_EDGE
    * ecc…

- Statistiche per BET_TAG:
    * BET_OVER15
    * BET_OVER25
    * BET_UNDER25
    * BET_UNDER15
    * ecc…

Esecuzione:
    python step5_backtest_over_under_signal.py --n 20000
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

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

SOFT_FILE = DATA_DIR / "step5_soft_history.parquet"
WIDE_FILE = DATA_DIR / "step4a_affini_index_wide_v2.parquet"

from app.ml.correlazioni_affini_v2.common.betting_rules.rule_over_signal import (
    rule_over_signal,
)


# --------------------------------------------------------------------
# EVALUATION BET TAG → BOOLEAN
# --------------------------------------------------------------------

def _eval_single_bet(
    tag: str,
    side: str,
    home_ft: int,
    away_ft: int,
) -> Optional[bool]:

    tg = home_ft + away_ft

    # UNDER globali
    if tag == "BET_UNDER15":
        return tg <= 1
    if tag == "BET_UNDER25":
        return tg <= 2

    # OVER globali
    if tag == "BET_OVER15":
        return tg >= 2
    if tag == "BET_OVER25":
        return tg >= 3

    return None


def _extract_raw_bet_tags_from_alert(alert) -> List[str]:
    raw_tags: List[str] = []

    meta = getattr(alert, "meta", {}) or {}
    bt = meta.get("bet_tags_raw")
    if isinstance(bt, list):
        for x in bt:
            if isinstance(x, str):
                raw_tags.append(x)
        if raw_tags:
            return raw_tags

    bets_field = getattr(alert, "bets", None) or []
    for b in bets_field:
        if isinstance(b, str):
            raw_tags.append(b)
        elif isinstance(b, dict):
            if "tag_raw" in b:
                raw_tags.append(b["tag_raw"])

    seen = set()
    deduped: List[str] = []
    for t in raw_tags:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return deduped


# --------------------------------------------------------------------
# BACKTEST
# --------------------------------------------------------------------

def run_backtest(n_matches: Optional[int] = None) -> None:
    print(f"📥 Carico soft_history da: {SOFT_FILE}")
    df_soft = pd.read_parquet(SOFT_FILE)

    print(f"📥 Carico wide da:        {WIDE_FILE}")
    wide = pd.read_parquet(WIDE_FILE)

    wide_cols_needed = [
        "match_id",
        "bk_p1", "bk_px", "bk_p2",
        "pic_p1", "pic_p2",
        "tightness_index",
        "lambda_total_form",
        "cluster_ou15",
        "cluster_ou25",
        "home_ft", "away_ft",
    ]
    missing = [c for c in wide_cols_needed if c not in wide.columns]
    if missing:
        raise RuntimeError(f"Mancano colonne nel wide: {missing}")

    wide_sel = wide[wide_cols_needed].copy()

    df = df_soft.merge(wide_sel, on="match_id", how="inner", suffixes=("", "_w"))
    df = df[df["status"] == "ok"].copy()

    if n_matches is not None and n_matches < len(df):
        df = df.sample(n_matches, random_state=42)
        print(f"✂️  Uso un sottoinsieme di {len(df)} partite.")
    else:
        print(f"➡️  Backtest su {len(df)} partite.")

    scenario_stats: Dict[str, Dict[str, float]] = {}
    bet_stats: Dict[str, Dict[str, float]] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), unit="match", desc="Backtesting"):

        soft_probs = {
            "pO15": float(row["soft_pO15"]),
            "pU15": float(row["soft_pU15"]),
            "pO25": float(row["soft_pO25"]),
            "pU25": float(row["soft_pU25"]),
        }
        ctx: Dict[str, Any] = {"soft_probs": soft_probs}

        alerts = rule_over_signal(row, ctx)
        if not alerts:
            continue

        alert = alerts[0]
        meta = alert.meta or {}

        scenario = meta.get("scenario", "UNKNOWN")
        side = "home"  # non rilevante per o/u, ma richiesto dal translator

        home_ft = int(row["home_ft"])
        away_ft = int(row["away_ft"])

        sc = scenario_stats.setdefault(scenario, {
            "n": 0,
            "over15": 0,
            "over25": 0,
        })

        sc["n"] += 1
        tg = home_ft + away_ft
        sc["over15"] += int(tg >= 2)
        sc["over25"] += int(tg >= 3)

        raw_tags = _extract_raw_bet_tags_from_alert(alert)

        for tag in raw_tags:
            res = _eval_single_bet(tag, side, home_ft, away_ft)
            if res is None:
                continue
            bstat = bet_stats.setdefault(tag, {"n": 0, "wins": 0})
            bstat["n"] += 1
            bstat["wins"] += int(res)

    # ----------------------------------------------------
    # REPORT
    # ----------------------------------------------------
    print("\n==============================================================")
    print("📊 RISULTATI BACKTEST OVER_UNDER_SIGNAL")
    print("==============================================================\n")

    print(f"Totale match analizzati:     {len(df)}")
    print()

    # SCENARI
    if scenario_stats:
        print("--------------------------------------------------------------")
        print("📌 Statistiche per SCENARIO")
        print("--------------------------------------------------------------")
        rows_scen = []
        for scen, s in scenario_stats.items():
            n = s["n"]
            rows_scen.append({
                "scenario": scen,
                "n": n,
                "over15_rate": s["over15"] / n,
                "over25_rate": s["over25"] / n,
            })

        df_scen = pd.DataFrame(rows_scen).sort_values("n", ascending=False)
        print(df_scen.to_string(index=False))
        print()

    # BET TAGS
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
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    run_backtest(n_matches=args.n)


if __name__ == "__main__":
    main()