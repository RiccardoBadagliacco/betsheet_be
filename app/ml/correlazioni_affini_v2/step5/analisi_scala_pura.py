#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5 — ANALISI MASTER SCALA PURA 1X2 / 2X1

Obiettivi:
    1) Usare:
        - step5_soft_history.parquet   (soft preds + risultato reale)
        - step4a_affini_index_wide_v2.parquet (quote bk/pic complete, feature estese)
    2) Definire SCALA PURA:
        - Scala pura 1X2:
            bk_p1  > bk_px  > bk_p2
            pic_p1 > pic_px > pic_p2

        - Scala pura 2X1:
            bk_p2  > bk_px  > bk_p1
            pic_p2 > pic_px > pic_p1

    3) Per ciascuna scala:
        - numero partite
        - P(1) / P(X) / P(2)
        - P(Over1.5), P(Over2.5), P(GG) se disponibili

    4) GRID SEARCH sulle probabilità:
        Lato 1:
            - regole del tipo:
                * pic_p1 >= t_pic
                * pic_p1 >= t_pic AND bk_p1 >= t_bk

        Lato 2:
            - simmetrico con pic_p2 / bk_p2

        Vincoli (parametrizzabili):
            - accuracy_min, accuracy_max
            - coverage_min
"""

import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# PATH
# -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR   = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR   = AFFINI_DIR / "data"

SOFT_FILE  = DATA_DIR / "step5_soft_history.parquet"
WIDE_FILE  = DATA_DIR / "step4a_affini_index_wide_v2.parquet"

OUT_RULES  = DATA_DIR / "step5_scala_pura_rules.parquet"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def safe_prop(x: pd.Series) -> float:
    if len(x) == 0:
        return float("nan")
    return float(x.mean())


def print_section(title: str):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)


# -------------------------------------------------------------------
# GRID SEARCH REGOLE — LATO 1 E LATO 2
# -------------------------------------------------------------------

def grid_search_scala_side(
    df: pd.DataFrame,
    side: str,
    acc_min: float = 0.75,
    acc_max: float = 0.90,
    cov_min: float = 0.20,
) -> pd.DataFrame:
    """
    Cerca regole semplici per massimizzare P(1) o P(2) in presenza di scala pura.

    side:
        - "1" → lavora su scala pura 1X2 e target is_home_win
        - "2" → lavora su scala pura 2X1 e target is_away_win

    Regole candidate:
        - solo picchetto:
            * pic_p1 >= t_pic   (lato 1)
            * pic_p2 >= t_pic   (lato 2)

        - combinazione pic + book:
            * pic_p1 >= t_pic AND bk_p1 >= t_bk
            * pic_p2 >= t_pic AND bk_p2 >= t_bk
    """

    df = df.copy()

    if side == "1":
        target_col = "is_home_win"
        pic_col    = "pic_p1"
        bk_col     = "bk_p1"
        scala_mask = df["scala_1x2"]
    elif side == "2":
        target_col = "is_away_win"
        pic_col    = "pic_p2"
        bk_col     = "bk_p2"
        scala_mask = df["scala_2x1"]
    else:
        raise ValueError("side deve essere '1' o '2'")

    # Sottoinsieme con scala pura lato scelto
    base = df[scala_mask].dropna(subset=[target_col, pic_col, bk_col]).copy()
    n_base = len(base)

    print_section(f"🔍 GRID SEARCH LATO {side} (n_base={n_base})")

    if n_base == 0:
        print("❌ Nessuna partita con scala pura su questo lato.")
        return pd.DataFrame()

    y = base[target_col].astype(int)

    # Range di soglie per pic / bk:
    pic_min = float(base[pic_col].quantile(0.30))
    pic_max = float(base[pic_col].quantile(0.98))
    bk_min  = float(base[bk_col].quantile(0.10))
    bk_max  = float(base[bk_col].quantile(0.98))

    pic_vals = np.linspace(pic_min, pic_max, 15)
    bk_vals  = np.linspace(bk_min, bk_max, 10)

    rules: List[Dict] = []

    # ----------------------------
    # 1) Regole solo picchetto
    # ----------------------------
    for t_pic in pic_vals:
        mask = base[pic_col] >= t_pic
        n_sel = mask.sum()
        if n_sel == 0:
            continue

        coverage = n_sel / n_base
        if coverage < cov_min:
            continue

        acc = safe_prop(y[mask])
        if not (acc_min <= acc <= acc_max):
            continue

        score = coverage * acc
        rules.append({
            "side": side,
            "rule_type": "pic_only",
            "rule": f"{pic_col} >= {t_pic:.3f}",
            "coverage": coverage,
            "accuracy": acc,
            "score": score,
            "n": n_sel,
        })

    # ----------------------------
    # 2) Regole pic + bk
    # ----------------------------
    for t_pic in pic_vals:
        for t_bk in bk_vals:
            mask = (base[pic_col] >= t_pic) & (base[bk_col] >= t_bk)
            n_sel = mask.sum()
            if n_sel == 0:
                continue

            coverage = n_sel / n_base
            if coverage < cov_min:
                continue

            acc = safe_prop(y[mask])
            if not (acc_min <= acc <= acc_max):
                continue

            score = coverage * acc
            rules.append({
                "side": side,
                "rule_type": "pic+bk",
                "rule": f"{pic_col} >= {t_pic:.3f} AND {bk_col} >= {t_bk:.3f}",
                "coverage": coverage,
                "accuracy": acc,
                "score": score,
                "n": n_sel,
            })

    if not rules:
        print("⚠️ Nessuna regola trovata con i vincoli richiesti.")
        return pd.DataFrame()

    res = pd.DataFrame(rules).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"✅ Regole trovate: {len(res)}")
    return res


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    print("====================================================")
    print("🚀 STEP5 — ANALISI MASTER SCALA PURA 1X2 / 2X1")
    print("====================================================")

    # 1) Caricamento file
    if not SOFT_FILE.exists():
        raise FileNotFoundError(f"Soft history non trovato: {SOFT_FILE}")
    if not WIDE_FILE.exists():
        raise FileNotFoundError(f"WIDE file non trovato: {WIDE_FILE}")

    soft = pd.read_parquet(SOFT_FILE)
    wide = pd.read_parquet(WIDE_FILE)

    print(f"📥 Soft history: {soft.shape}")
    print(f"📥 WIDE index : {wide.shape}")

    # 2) Merge su match_id
    if "match_id" not in soft.columns or "match_id" not in wide.columns:
        raise RuntimeError("❌ match_id mancante in soft o wide.")

    df = pd.merge(
        soft[["match_id", "is_over15_real", "is_over25_real", "is_gg_real"]],
        wide[
            [
                "match_id",
                "bk_p1", "bk_px", "bk_p2",
                "pic_p1", "pic_px", "pic_p2",
                "is_home_win", "is_draw", "is_away_win",
            ]
        ],
        on="match_id",
        how="inner",
    )

    print(f"📏 Merge finale: {df.shape}")

    # 3) Definizione SCALA PURA
    df["scala_1x2"] = (
        (df["bk_p1"]  > df["bk_px"]) &
        (df["bk_px"]  > df["bk_p2"]) &
        (df["pic_p1"] > df["pic_px"]) &
        (df["pic_px"] > df["pic_p2"])
    )

    df["scala_2x1"] = (
        (df["bk_p2"]  > df["bk_px"]) &
        (df["bk_px"]  > df["bk_p1"]) &
        (df["pic_p2"] > df["pic_px"]) &
        (df["pic_px"] > df["pic_p1"])
    )

    n_total   = len(df)
    n_scala_1 = int(df["scala_1x2"].sum())
    n_scala_2 = int(df["scala_2x1"].sum())

    print_section("📊 CONTEGGI SCALA PURA")
    print(f"Totale match: {n_total}")
    print(f"Scala pura 1X2: {n_scala_1} ({n_scala_1 / n_total:.3%})")
    print(f"Scala pura 2X1: {n_scala_2} ({n_scala_2 / n_total:.3%})")

    # 4) Statistiche di base per ogni scala
    print_section("📊 STATISTICHE SCALA PURA 1X2 (lato 1)")
    df1 = df[df["scala_1x2"]].copy()
    if not df1.empty:
        print(f"Match: {len(df1)}")
        print(f"P(1) = {df1['is_home_win'].mean():.3f}")
        print(f"P(X) = {df1['is_draw'].mean():.3f}")
        print(f"P(2) = {df1['is_away_win'].mean():.3f}")
        if "is_over15_real" in df1:
            print(f"P(Over 1.5) = {df1['is_over15_real'].mean():.3f}")
        if "is_over25_real" in df1:
            print(f"P(Over 2.5) = {df1['is_over25_real'].mean():.3f}")
        if "is_gg_real" in df1:
            print(f"P(GG) = {df1['is_gg_real'].mean():.3f}")
    else:
        print("Nessuna partita con scala 1X2.")

    print_section("📊 STATISTICHE SCALA PURA 2X1 (lato 2)")
    df2 = df[df["scala_2x1"]].copy()
    if not df2.empty:
        print(f"Match: {len(df2)}")
        print(f"P(1) = {df2['is_home_win'].mean():.3f}")
        print(f"P(X) = {df2['is_draw'].mean():.3f}")
        print(f"P(2) = {df2['is_away_win'].mean():.3f}")
        if "is_over15_real" in df2:
            print(f"P(Over 1.5) = {df2['is_over15_real'].mean():.3f}")
        if "is_over25_real" in df2:
            print(f"P(Over 2.5) = {df2['is_over25_real'].mean():.3f}")
        if "is_gg_real" in df2:
            print(f"P(GG) = {df2['is_gg_real'].mean():.3f}")
    else:
        print("Nessuna partita con scala 2X1.")

    # 5) GRID SEARCH per regole lato 1 e lato 2
    print_section("🔧 GRID SEARCH SCALA PURA — LATO 1 (P(1))")
    rules_1 = grid_search_scala_side(df, side="1")

    print_section("🔧 GRID SEARCH SCALA PURA — LATO 2 (P(2))")
    rules_2 = grid_search_scala_side(df, side="2")

    # 6) Output finale
    all_rules = pd.concat(
        [r for r in [rules_1, rules_2] if r is not None and not r.empty],
        ignore_index=True
    ) if ((rules_1 is not None and not rules_1.empty) or (rules_2 is not None and not rules_2.empty)) else pd.DataFrame()

    if all_rules.empty:
        print("⚠️ Nessuna regola scala pura salvata (nessun match ai vincoli).")
    else:
        print_section("🏆 TOP 20 REGOLE GLOBALI SCALA PURA")
        print(all_rules.sort_values("score", ascending=False).head(20))

        OUT_RULES.parent.mkdir(parents=True, exist_ok=True)
        all_rules.to_parquet(OUT_RULES, index=False)
        print(f"\n💾 Regole scala pura salvate in: {OUT_RULES}")

    print("\n====================================================")
    print("🏁 ANALISI MASTER SCALA PURA COMPLETATA")
    print("====================================================")


if __name__ == "__main__":
    main()