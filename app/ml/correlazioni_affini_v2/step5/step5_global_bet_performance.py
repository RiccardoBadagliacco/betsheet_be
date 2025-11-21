#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5 — GLOBAL BET PERFORMANCE

Obiettivo:
- Per ogni match storico con soft_probs e dati wide:
    * costruire soft_probs
    * applicare TUTTE le regole registrate
        - rule_favorite_profile_signal
        - rule_over_signal
        - (eventuali altre future)
    * estrarre i bet_tags suggeriti
    * valutarli rispetto al risultato reale con UNICO _eval_bet

Output:
- Statistiche GLOBALI per bet_tag:
    * n, wins, win_rate

- Statistiche per REGOLA e bet_tag:
    * rule_name -> bet_tag -> {n, wins, win_rate}

Salvato in:
    data/step5_bet_global_stats.json

Esecuzione:
    python step5_global_bet_performance.py --n 20000
"""

import sys
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from tqdm import tqdm

# -----------------------
# PATH & IMPORT
# -----------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = Path(__file__).resolve().parents[1]  # correlazioni_affini_v2
DATA_DIR = BASE_DIR / "data"

SOFT_FILE = DATA_DIR / "step5_soft_history.parquet"
WIDE_FILE = DATA_DIR / "step4a_affini_index_wide_v2.parquet"
OUT_FILE  = DATA_DIR / "step5_bet_global_stats.json"

# Regole disponibili
from app.ml.correlazioni_affini_v2.common.betting_rules.rule_favorite_profile_signal import (
    rule_favorite_profile_signal,
)
from app.ml.correlazioni_affini_v2.common.betting_rules.rule_over_signal import (
    rule_over_signal,
)


# Registro delle regole da applicare
REGISTERED_RULES: List[Tuple[str, Any]] = [
    ("FAVORITE_PROFILE_SIGNAL", rule_favorite_profile_signal),
    ("OVER_UNDER_SIGNAL",       rule_over_signal),
    # Qui in futuro puoi aggiungere altre regole:
    # ("ALTRA_REGOLA", rule_altra_regola),
]


# --------------------------------------------------------------------
# EVAL BET UNIFICATO
# --------------------------------------------------------------------

def _eval_bet(
    tag: str,
    side: str,
    home_ft: int,
    away_ft: int,
) -> Optional[bool]:
    """
    Valutazione unificata di TUTTI i bet_tag possibili.

    Ritorna:
        True  -> bet vinta
        False -> bet persa
        None  -> bet non valutabile (es. NO_BET, Asian -0.25, tag ignoto)
    """
    tg = home_ft + away_ft

    # Determino gol favorita / avversaria dove serve
    if side == "away":
        fav_goals = away_ft
        opp_goals = home_ft
    else:
        # default: side="home" o qualunque altro valore → trattiamo "home" come favorita
        fav_goals = home_ft
        opp_goals = away_ft

    # -------------------------
    # Tag gestionali / non betting
    # -------------------------
    if tag in {"BET_NO_BET_STRONG"}:
        return None

    # -------------------------
    # UNDER / OVER GLOBALI
    # -------------------------
    if tag == "BET_UNDER15":
        return tg <= 1
    if tag == "BET_UNDER25":
        return tg <= 2

    if tag == "BET_OVER15":
        return tg >= 2
    if tag == "BET_OVER25":
        return tg >= 3

    # -------------------------
    # Segni fissi
    # -------------------------
    if tag == "BET_1":
        return home_ft > away_ft
    if tag == "BET_2":
        return away_ft > home_ft

    # -------------------------
    # Doppie chance
    # -------------------------
    if tag == "BET_DC_OPPOSITE":
        # DC contro favorita:
        #  - se favorita è home -> X2 (vinta se home non vince)
        #  - se favorita è away -> 1X (vinta se away non vince)
        if side == "home":
            return home_ft <= away_ft  # X o 2
        else:
            return away_ft <= home_ft  # X o 1

    if tag == "BET_DC_FAV":
        # DC a favore favorita:
        #  - se favorita è home -> 1X (vinta se home non perde)
        #  - se favorita è away -> X2 (vinta se away non perde)
        if side == "home":
            return home_ft >= away_ft  # 1 o X
        else:
            return away_ft >= home_ft  # 2 o X

    # -------------------------
    # Gol squadra favorita
    # -------------------------
    if tag in {"BET_FAV_SEGNA", "BET_FAV_OVER_0_5"}:
        return fav_goals >= 1

    if tag == "BET_LAY_FAV_SEGNA":
        # banca "favorita segna"
        return fav_goals == 0

    if tag == "BET_UNDER_1_5_FAV":
        return fav_goals <= 1

    # -------------------------
    # Multigol favorita
    # -------------------------
    if tag == "BET_FAV_1_3":
        return 1 <= fav_goals <= 3
    if tag == "BET_FAV_1_4":
        return 1 <= fav_goals <= 4
    if tag == "BET_FAV_1_5":
        return 1 <= fav_goals <= 5

    # -------------------------
    # Asian vari (per ora NON valutati)
    # -------------------------
    if tag in {"BET_ASIAN_-0_25_FAV"}:
        return None

    # -------------------------
    # Tag sconosciuto → non valutabile
    # -------------------------
    return None


# --------------------------------------------------------------------
# HELPER: estrazione bet_tags da alert
# --------------------------------------------------------------------

def _extract_raw_bet_tags_from_alert(alert) -> List[str]:
    """
    Resiliente a variazioni nel modello BettingAlert.

    Supporta:
    - meta["bet_tags_raw"] = ["BET_...", ...]
    - alert.bets = ["BET_...", ...]
    - alert.bets = [{"tag_raw": "...", ...}, ...]
    """
    raw_tags: List[str] = []

    meta = getattr(alert, "meta", {}) or {}

    # 1) meta["bet_tags_raw"] (sorgente "ufficiale")
    bt = meta.get("bet_tags_raw")
    if isinstance(bt, list):
        for x in bt:
            if isinstance(x, str):
                raw_tags.append(x)
        if raw_tags:
            # deduplica mantenendo ordine
            seen = set()
            deduped: List[str] = []
            for t in raw_tags:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)
            return deduped

    # 2) fallback: alert.bets
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
# FUNZIONE PRINCIPALE
# --------------------------------------------------------------------

def run_analysis(n_matches: Optional[int] = None) -> None:
    # ---------------------------
    # 1. Carica soft_history + wide
    # ---------------------------
    print("====================================================")
    print("🚀 STEP5 — GLOBAL BET PERFORMANCE")
    print("====================================================\n")

    print(f"📥 Carico soft_history da: {SOFT_FILE}")
    df_soft = pd.read_parquet(SOFT_FILE)

    print(f"📥 Carico wide da:        {WIDE_FILE}")
    wide = pd.read_parquet(WIDE_FILE)

    # Colonne che servono alle varie regole e alla valutazione
    wide_cols_needed = [
        "match_id",
        "bk_p1", "bk_px", "bk_p2",
        "pic_p1", "pic_p2",
        "tightness_index",
        "lambda_total_form",
        "cluster_1x2",
        "cluster_ou15",
        "cluster_ou25",
        "home_ft", "away_ft",
    ]
    missing = [c for c in wide_cols_needed if c not in wide.columns]
    if missing:
        raise RuntimeError(f"Mancano colonne nel wide: {missing}")

    wide_sel = wide[wide_cols_needed].copy()

    # Merge soft_history + wide
    df = df_soft.merge(wide_sel, on="match_id", how="inner", suffixes=("", "_w"))

    # Filtra solo status ok
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    if n_matches is not None and n_matches < len(df):
        df = df.sample(n_matches, random_state=42).copy()
        print(f"✂️  Uso un sottoinsieme di {len(df)} partite per l'analisi.")
    else:
        print(f"➡️  Analisi su tutte le partite disponibili: {len(df)}")

    # ---------------------------
    # 2. Accumulatori statistiche
    # ---------------------------
    # global_stats: bet_tag -> {n, wins}
    global_stats: Dict[str, Dict[str, int]] = {}

    # by_rule_stats: rule_name -> bet_tag -> {n, wins}
    by_rule_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

    # ---------------------------
    # 3. Loop sui match
    # ---------------------------
    for _, row in tqdm(df.iterrows(), total=len(df), unit="match", desc="Analisi"):
        home_ft = int(row["home_ft"])
        away_ft = int(row["away_ft"])

        # soft_probs completi (1X2 + O/U 1.5 + O/U 2.5)
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

        # calcolo favorita "tecnica" come fallback
        try:
            bk_p1 = float(row["bk_p1"])
            bk_px = float(row["bk_px"])
            bk_p2 = float(row["bk_p2"])
        except Exception:
            bk_p1 = bk_px = bk_p2 = math.nan

        is_home_fav = (not math.isnan(bk_p1)) and bk_p1 > max(bk_px, bk_p2)
        is_away_fav = (not math.isnan(bk_p2)) and bk_p2 > max(bk_px, bk_p1)

        default_side = "home"
        if is_home_fav:
            default_side = "home"
        elif is_away_fav:
            default_side = "away"

        # ---------------------------
        # 3a. Applica tutte le regole registrate
        # ---------------------------
        for rule_name, rule_fn in REGISTERED_RULES:
            try:
                alerts = rule_fn(row, ctx)
            except Exception as e:
                print(f"⚠️  ERRORE in regola {rule_name}: {e}")
                continue

            if not alerts:
                continue

            # Di solito una sola alert, ma per sicurezza looppiamo
            for alert in alerts:
                meta = alert.meta or {}

                side = meta.get("side")
                if side not in {"home", "away"}:
                    side = default_side

                raw_tags = _extract_raw_bet_tags_from_alert(alert)
                if not raw_tags:
                    continue

                # dedup per regola-match-alert
                raw_tags = list(dict.fromkeys(raw_tags))

                # Aggiorna stats per ogni bet_tag
                for tag in raw_tags:
                    res = _eval_bet(tag, side, home_ft, away_ft)
                    if res is None:
                        continue

                    # GLOBAL
                    g = global_stats.setdefault(tag, {"n": 0, "wins": 0})
                    g["n"] += 1
                    g["wins"] += int(res)

                    # PER REGOLA
                    rstats = by_rule_stats.setdefault(rule_name, {})
                    br = rstats.setdefault(tag, {"n": 0, "wins": 0})
                    br["n"] += 1
                    br["wins"] += int(res)

    # ---------------------------
    # 4. Costruzione output con win_rate
    # ---------------------------
    global_out: Dict[str, Dict[str, float]] = {}
    for tag, s in global_stats.items():
        n = s["n"]
        wins = s["wins"]
        win_rate = wins / n if n > 0 else math.nan
        global_out[tag] = {
            "n": int(n),
            "wins": int(wins),
            "win_rate": float(win_rate),
        }

    by_rule_out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rule_name, tags_dict in by_rule_stats.items():
        out_tags: Dict[str, Dict[str, float]] = {}
        for tag, s in tags_dict.items():
            n = s["n"]
            wins = s["wins"]
            win_rate = wins / n if n > 0 else math.nan
            out_tags[tag] = {
                "n": int(n),
                "wins": int(wins),
                "win_rate": float(win_rate),
            }
        by_rule_out[rule_name] = out_tags

    result = {
        "global": global_out,
        "by_rule": by_rule_out,
        "meta": {
            "n_matches": int(len(df)),
        },
    }

    # ---------------------------
    # 5. Salvataggio JSON (RISCRITTO)
    # ---------------------------
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ---------------------------
    # 6. Report riassuntivo
    # ---------------------------
    print("\n==============================================================")
    print("📊 GLOBAL BET PERFORMANCE")
    print("==============================================================\n")

    print(f"Totale match analizzati: {len(df)}")
    print(f"File JSON scritto in:    {OUT_FILE}\n")

    if global_out:
        print("--------------------------------------------------------------")
        print("🎯 Statistiche GLOBALI per BET_TAG (ordinate per n)")
        print("--------------------------------------------------------------")
        rows = []
        for tag, s in global_out.items():
            rows.append({
                "bet_tag": tag,
                "n": s["n"],
                "win_rate": s["win_rate"],
            })
        df_global = pd.DataFrame(rows).sort_values("n", ascending=False)
        print(df_global.to_string(index=False))
        print()

    if by_rule_out:
        print("--------------------------------------------------------------")
        print("📌 Statistiche per REGOLA e BET_TAG")
        print("--------------------------------------------------------------")
        for rule_name, tags_dict in by_rule_out.items():
            print(f"\n▶️  Regola: {rule_name}")
            rows_r = []
            for tag, s in tags_dict.items():
                rows_r.append({
                    "bet_tag": tag,
                    "n": s["n"],
                    "win_rate": s["win_rate"],
                })
            df_rule = pd.DataFrame(rows_r).sort_values("n", ascending=False)
            print(df_rule.to_string(index=False))
            print()

    print("🏁 ANALISI COMPLETATA.")


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
    run_analysis(n_matches=args.n)


if __name__ == "__main__":
    main()
