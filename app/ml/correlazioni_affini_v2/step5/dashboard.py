#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dashboard avanzata per giocare con QUALSIASI combinazione di regole
su tutti gli attributi (soft_probs + feature SLIM) e vedere:

- % reali di 1 / X / 2
- % reali di Over 1.5 / Over 2.5
- % reali di GG / NoGG

Regole definite come:
  (colonna, operatore, valore) combinate con AND oppure OR
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------
# PATH e caricamento dati
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"

SOFT_FILE = DATA_DIR / "step5_soft_history.parquet"
SLIM_FILE = DATA_DIR / "step4b_affini_index_slim_v2.parquet"


@st.cache_data
def load_merged():
    soft = pd.read_parquet(SOFT_FILE)
    slim = pd.read_parquet(SLIM_FILE)

    # merge 1:1 su match_id (soft ha solo match giocati)
    df = soft.merge(slim, on="match_id", how="left", suffixes=("", "_slim"))

    # colonne derivate per esiti reali (safety, anche se già presenti)
    if "home_ft" in df.columns and "away_ft" in df.columns:
        gh = df["home_ft"].astype(int)
        ga = df["away_ft"].astype(int)
        tg = gh + ga

        df["is_home_win_real"] = (gh > ga).astype(int)
        df["is_draw_real"] = (gh == ga).astype(int)
        df["is_away_win_real"] = (gh < ga).astype(int)

        df["is_over15_real"] = (tg >= 2).astype(int)
        df["is_over25_real"] = (tg >= 3).astype(int)
        df["is_gg_real"] = ((gh > 0) & (ga > 0)).astype(int)
        df["is_nogg_real"] = 1 - df["is_gg_real"]

    return df


df_all = load_merged()

st.set_page_config(
    page_title="Soft Engine – Rule Lab",
    layout="wide",
)

st.title("🧪 Soft Engine – Rule Lab Avanzato")
st.caption(
    "Gioca con QUALSIASI combinazione di regole su tutti gli attributi "
    "per vedere le frequenze reali di 1X2, Over/Under e GG/NoGG."
)

# ---------------------------------------------------
# UTILS
# ---------------------------------------------------
def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def apply_rule_builder(df: pd.DataFrame, rules: list[dict], combinator: str) -> pd.DataFrame:
    if not rules:
        return df

    masks = []
    for r in rules:
        col = r["col"]
        op = r["op"]
        val = r.get("val")
        val2 = r.get("val2")
        vals = r.get("vals")

        if col not in df.columns:
            continue

        s = df[col]

        # costruisco mask in base all'operatore
        if op == "==":
            mask = s == val
        elif op == "!=":
            mask = s != val
        elif op == ">":
            mask = s > val
        elif op == ">=":
            mask = s >= val
        elif op == "<":
            mask = s < val
        elif op == "<=":
            mask = s <= val
        elif op == "between":
            mask = s.between(val, val2)
        elif op == "in":
            if not vals:
                # niente selezionato => nessun filtraggio su questa regola
                mask = pd.Series(True, index=df.index)
            else:
                mask = s.isin(vals)
        else:
            # operatore sconosciuto → ignoro
            continue

        masks.append(mask)

    if not masks:
        return df

    if combinator == "AND":
        full_mask = masks[0]
        for m in masks[1:]:
            full_mask &= m
    else:  # OR
        full_mask = masks[0]
        for m in masks[1:]:
            full_mask |= m

    return df[full_mask].copy()


# ---------------------------------------------------
# SIDEBAR – RULE BUILDER
# ---------------------------------------------------
st.sidebar.header("⚙️ Rule Builder")

st.sidebar.markdown(
    "Qui definisci le regole del tipo:\n"
    "`colonna OP valore` combinate con **AND** oppure **OR**."
)

# lista colonne candidate (escludiamo giusto qualche colonna di servizio)
exclude_cols = {"status", "reason"}
all_cols = [c for c in df_all.columns if c not in exclude_cols]

# selezione numero di regole
n_rules = st.sidebar.number_input("Numero di condizioni", min_value=1, max_value=12, value=3, step=1)

combinator = st.sidebar.radio(
    "Combina le condizioni con:",
    ["AND", "OR"],
    horizontal=True,
)

rules = []
for i in range(int(n_rules)):
    st.sidebar.markdown(f"#### Regola {i+1}")

    col = st.sidebar.selectbox(
        f"Colonna {i+1}",
        options=["(disattiva)"] + all_cols,
        key=f"col_{i}",
    )

    if col == "(disattiva)":
        continue

    series = df_all[col]
    numeric = is_numeric(series)

    if numeric:
        ops = ["between", ">=", "<=", ">", "<", "==", "!="]
    else:
        ops = ["in", "==", "!="]

    op = st.sidebar.selectbox(
        f"Operatore {i+1}",
        options=ops,
        key=f"op_{i}",
    )

    rule = {"col": col, "op": op}

    if numeric:
        col_min = float(np.nanmin(series))
        col_max = float(np.nanmax(series))
        # un minimo padding
        span = col_max - col_min
        col_min -= 0.02 * span
        col_max += 0.02 * span

        if op == "between":
            v1, v2 = st.sidebar.slider(
                f"Valore (range) {i+1}",
                min_value=float(col_min),
                max_value=float(col_max),
                value=(float(col_min), float(col_max)),
                key=f"val_{i}",
            )
            rule["val"] = v1
            rule["val2"] = v2
        else:
            v = st.sidebar.number_input(
                f"Valore {i+1}",
                value=float(col_min),
                key=f"val_{i}",
            )
            rule["val"] = v
    else:
        # categoriale
        uniques = series.dropna().unique().tolist()
        uniques = sorted(uniques)

        if op == "in":
            vals = st.sidebar.multiselect(
                f"Valori {i+1}",
                options=uniques,
                default=[],
                key=f"vals_{i}",
            )
            rule["vals"] = vals
        else:
            if uniques:
                v = st.sidebar.selectbox(
                    f"Valore {i+1}",
                    options=uniques,
                    key=f"val_{i}",
                )
                rule["val"] = v
            else:
                rule["val"] = None

    rules.append(rule)

# ---------------------------------------------------
# APPLY RULES
# ---------------------------------------------------
df_filtered = apply_rule_builder(df_all, rules, combinator)

st.subheader("📌 Risultati del filtro")
st.write(f"Match trovati: **{len(df_filtered)}**")

if len(df_filtered) == 0:
    st.warning("Nessun match soddisfa le regole impostate. Modifica le condizioni nella sidebar.")
    st.stop()

# ---------------------------------------------------
# CALCOLO STATISTICHE 1X2, OU, GG
# ---------------------------------------------------
def safe_mean(s):
    if s.isna().all():
        return np.nan
    return s.mean()


# Real 1X2
p1_real = safe_mean(df_filtered["is_home_win_real"])
px_real = safe_mean(df_filtered["is_draw_real"])
p2_real = safe_mean(df_filtered["is_away_win_real"])

# Real OU & GG
pO15_real = safe_mean(df_filtered["is_over15_real"])
pO25_real = safe_mean(df_filtered["is_over25_real"])
pGG_real  = safe_mean(df_filtered["is_gg_real"])
pNoGG_real = safe_mean(df_filtered["is_nogg_real"])

# Soft probs medie
soft_cols = ["soft_p1", "soft_px", "soft_p2", "soft_pO15", "soft_pO25", "soft_pU15", "soft_pU25"]
soft_means = {c: safe_mean(df_filtered[c]) for c in soft_cols if c in df_filtered.columns}

# ---------------------------------------------------
# DISPLAY METRICS
# ---------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Real P(1)", f"{p1_real:.3f}" if pd.notna(p1_real) else "N/A")
c2.metric("Real P(X)", f"{px_real:.3f}" if pd.notna(px_real) else "N/A")
c3.metric("Real P(2)", f"{p2_real:.3f}" if pd.notna(p2_real) else "N/A")

c4, c5, c6 = st.columns(3)
c4.metric("Real P(Over 1.5)", f"{pO15_real:.3f}" if pd.notna(pO15_real) else "N/A")
c5.metric("Real P(Over 2.5)", f"{pO25_real:.3f}" if pd.notna(pO25_real) else "N/A")
c6.metric("Real P(GG)", f"{pGG_real:.3f}" if pd.notna(pGG_real) else "N/A")

c7, _, c9 = st.columns(3)
c7.metric("Real P(NoGG)", f"{pNoGG_real:.3f}" if pd.notna(pNoGG_real) else "N/A")
if "soft_pO15" in soft_means:
    c9.metric("Soft P(Over 1.5)", f"{soft_means['soft_pO15']:.3f}")

st.markdown("---")

st.subheader("📊 Soft vs Real (medie)")

cols_soft = st.columns(3)
if "soft_p1" in soft_means:
    cols_soft[0].metric("Soft P(1)", f"{soft_means['soft_p1']:.3f}")
if "soft_pO25" in soft_means:
    cols_soft[1].metric("Soft P(O2.5)", f"{soft_means['soft_pO25']:.3f}")
if "soft_pU25" in soft_means:
    cols_soft[2].metric("Soft P(U2.5)", f"{soft_means['soft_pU25']:.3f}")

# ---------------------------------------------------
# TABELLA DETTAGLIATA
# ---------------------------------------------------
st.markdown("### 📄 Match dettagliati (prime 300 righe)")

cols_default = [
    "date", "league", "home_team", "away_team",
    "soft_p1", "soft_px", "soft_p2",
    "soft_pO15", "soft_pO25",
    "home_ft", "away_ft", "total_goals",
    "is_over15_real", "is_over25_real", "is_gg_real",
    "cluster_1x2", "cluster_ou25", "cluster_ou15",
]

cols_show = [c for c in cols_default if c in df_filtered.columns]
df_show = df_filtered[cols_show].head(300)

st.dataframe(df_show)