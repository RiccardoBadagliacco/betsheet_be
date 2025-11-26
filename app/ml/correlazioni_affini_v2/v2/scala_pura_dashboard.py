# app/ml/correlazioni_affini_v2/v2/scala_pura_dashboard.py

import sys
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
import streamlit as st

ScalaType = Literal["1X2", "2X1"]

REPO_ROOT = Path(__file__).resolve().parents[4]


# -------------------------------------------------------------------
# Path helper (come negli altri script)
# -------------------------------------------------------------------
def _resolve(names: List[str]) -> Path:
    script_dir = Path(__file__).resolve().parent
    local_data_dir = script_dir / "data"
    common_data_dir = script_dir.parent / "data"
    root_data_dir = REPO_ROOT / "data"

    for name in names:
        for base in (local_data_dir, common_data_dir, root_data_dir):
            candidate = base / name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"File {names} non trovato. Cercato in {local_data_dir}, {common_data_dir}, {root_data_dir}"
    )


# -------------------------------------------------------------------
# Stesse funzioni di scala_pura_analysis (duplicate per indipendenza)
# -------------------------------------------------------------------
def add_scala_pura_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["pic_p1", "pic_px", "pic_p2", "bk_p1", "bk_px", "bk_p2"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"add_scala_pura_flags: manca colonna {c}")

    pic_p1 = df["pic_p1"].astype(float)
    pic_px = df["pic_px"].astype(float)
    pic_p2 = df["pic_p2"].astype(float)
    bk_p1 = df["bk_p1"].astype(float)
    bk_px = df["bk_px"].astype(float)
    bk_p2 = df["bk_p2"].astype(float)

    cond_model_1x2 = (pic_p1 > pic_px) & (pic_px > pic_p2)
    cond_book_1x2 = (bk_p1 > bk_px) & (bk_px > bk_p2)
    is_scala_1x2 = cond_model_1x2 & cond_book_1x2

    cond_model_2x1 = (pic_p2 > pic_px) & (pic_px > pic_p1)
    cond_book_2x1 = (bk_p2 > bk_px) & (bk_px > bk_p1)
    is_scala_2x1 = cond_model_2x1 & cond_book_2x1

    df["is_scala_1x2"] = is_scala_1x2
    df["is_scala_2x1"] = is_scala_2x1

    df["scala_type"] = None
    df.loc[is_scala_1x2, "scala_type"] = "1X2"
    df.loc[is_scala_2x1, "scala_type"] = "2X1"
    return df


def add_mg_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "home_ft" not in df.columns or "away_ft" not in df.columns:
        raise RuntimeError("add_mg_columns: servono home_ft e away_ft")

    h = df["home_ft"].astype(float)
    a = df["away_ft"].astype(float)

    df["mg_home_1_3"] = (h >= 1) & (h <= 3)
    df["mg_home_1_4"] = (h >= 1) & (h <= 4)
    df["mg_home_1_5"] = (h >= 1) & (h <= 5)

    df["mg_away_1_3"] = (a >= 1) & (a <= 3)
    df["mg_away_1_4"] = (a >= 1) & (a <= 4)
    df["mg_away_1_5"] = (a >= 1) & (a <= 5)

    return df


def build_odds_buckets(
    fav_odds: pd.Series,
    low: float = 1.01,
    high: float = 2.50,
    step: float = 0.10,
) -> Tuple[pd.Series, list, np.ndarray]:
    edges = np.arange(low, high + step, step)
    labels = []
    for i in range(len(edges) - 1):
        l = edges[i]
        r = edges[i + 1]
        labels.append(f"{l:.2f}-{r:.2f}")

    bucket = pd.cut(fav_odds, bins=edges, labels=labels, right=False, include_lowest=True)
    return bucket, labels, edges


def compute_roi(outcome: pd.Series, odds: pd.Series) -> float:
    mask = outcome.astype(bool)
    n = len(outcome)
    if n == 0:
        return np.nan
    winnings = odds[mask].fillna(0).sum()
    roi = (winnings - n) / n
    return float(roi)


def compute_scala_stats(df: pd.DataFrame, scala_type: ScalaType) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    if scala_type == "1X2":
        mask = df["is_scala_1x2"].astype(bool)
        fav_odds = df["avg_home_odds"].astype(float)
        outcome_main = df["is_home_win"].astype(bool)
    else:
        mask = df["is_scala_2x1"].astype(bool)
        fav_odds = df["avg_away_odds"].astype(float)
        outcome_main = df["is_away_win"].astype(bool)

    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame(), sub

    sub["fav_odds"] = fav_odds[mask]
    sub["odds_bucket"], labels, edges = build_odds_buckets(sub["fav_odds"])

    rows = []
    for label in labels:
        g = sub[sub["odds_bucket"] == label]
        n = len(g)
        if n == 0:
            continue

        p1 = g["is_home_win"].mean()
        pX = g["is_draw"].mean()
        p2 = g["is_away_win"].mean()

        freq_1X = ((g["is_home_win"].astype(bool)) | (g["is_draw"].astype(bool))).mean()
        freq_X2 = ((g["is_draw"].astype(bool)) | (g["is_away_win"].astype(bool))).mean()
        freq_12 = ((g["is_home_win"].astype(bool)) | (g["is_away_win"].astype(bool))).mean()

        if scala_type == "1X2":
            odds_main = g["avg_home_odds"].astype(float)
        else:
            odds_main = g["avg_away_odds"].astype(float)
        roi_main = compute_roi(outcome_main[g.index], odds_main)

        p_over15 = g["is_over15"].mean() if "is_over15" in g.columns else np.nan
        p_under15 = 1.0 - p_over15 if p_over15 == p_over15 else np.nan

        p_over25 = g["is_over25"].mean() if "is_over25" in g.columns else np.nan
        if "is_under25" in g.columns:
            p_under25 = g["is_under25"].mean()
        else:
            p_under25 = 1.0 - p_over25 if p_over25 == p_over25 else np.nan

        p_home_1_3 = g["mg_home_1_3"].mean()
        p_home_1_4 = g["mg_home_1_4"].mean()
        p_home_1_5 = g["mg_home_1_5"].mean()

        p_away_1_3 = g["mg_away_1_3"].mean()
        p_away_1_4 = g["mg_away_1_4"].mean()
        p_away_1_5 = g["mg_away_1_5"].mean()

        l_float = float(label.split("-")[0])
        r_float = float(label.split("-")[1])
        bucket_mid = 0.5 * (l_float + r_float)

        rows.append(
            {
                "scala_type": scala_type,
                "odds_bucket": label,
                "bucket_mid": bucket_mid,
                "n_matches": n,
                "winrate_main_sign": outcome_main[g.index].mean(),
                "roi_main_sign": roi_main,
                "p1": p1,
                "pX": pX,
                "p2": p2,
                "freq_1X": freq_1X,
                "freq_X2": freq_X2,
                "freq_12": freq_12,
                "p_over15": p_over15,
                "p_under15": p_under15,
                "p_over25": p_over25,
                "p_under25": p_under25,
                "p_home_1_3": p_home_1_3,
                "p_home_1_4": p_home_1_4,
                "p_home_1_5": p_home_1_5,
                "p_away_1_3": p_away_1_3,
                "p_away_1_4": p_away_1_4,
                "p_away_1_5": p_away_1_5,
            }
        )

    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats = stats.sort_values("bucket_mid")
    return stats, sub


# -------------------------------------------------------------------
# Caricamento dati (con cache)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_datamaster() -> pd.DataFrame:
    dm_path = _resolve(["datamaster_1x2.parquet"])
    df = pd.read_parquet(dm_path)
    df = add_scala_pura_flags(df)
    df = add_mg_columns(df)
    return df


# -------------------------------------------------------------------
# UI Tab singolo per una Scala
# -------------------------------------------------------------------
def render_scala_tab(df: pd.DataFrame, scala_type: ScalaType):
    stats, sub = compute_scala_stats(df, scala_type)

    if sub.empty:
        st.warning("Nessun match che rispetta la Scala Pura per questa tipologia.")
        return

    # -------------------------------
    # Range quota favorita (nuovo slider interattivo)
    # -------------------------------
    if scala_type == "1X2":
        fav_all = df[df["is_scala_1x2"]]["avg_home_odds"].astype(float)
    else:
        fav_all = df[df["is_scala_2x1"]]["avg_away_odds"].astype(float)

    if len(fav_all) > 0:
        global_min = float(fav_all.min())
        global_max = float(fav_all.max())
    else:
        global_min, global_max = 1.01, 3.50

    st.markdown("### Range quota favorita")
    min_q, max_q = st.slider(
        "Seleziona range quota favorita",
        min_value=global_min,
        max_value=global_max,
        value=(global_min, global_max),
        step=0.01,
        key=f"slider_range_{scala_type}"
    )

    # Applica filtro range quote
    sub = sub[(sub["fav_odds"] >= min_q) & (sub["fav_odds"] <= max_q)]
    stats = stats[(stats["bucket_mid"] >= min_q) & (stats["bucket_mid"] <= max_q)]

    # -------------------------------
    # PATCH: filtro Picchetto > Book favorita
    # -------------------------------
    st.markdown("### Confronto Picchetto vs Book sulla favorita")
    apply_pic_vs_book = st.checkbox(
        "Mostra SOLO match dove la quota del picchetto della favorita è > quota del book",
        key=f"pic_vs_book_{scala_type}"
    )

    if apply_pic_vs_book:
        if scala_type == "1X2":
            # favorita = casa
            sub = sub[sub["pic_odd1"] > sub["avg_home_odds"]]
        else:
            # favorita = ospite
            sub = sub[sub["pic_odd2"] > sub["avg_away_odds"]]

        # Aggiorna stats in base ai bucket effettivamente presenti
        if not sub.empty and "odds_bucket" in sub.columns:
            valid_buckets = sub["odds_bucket"].unique().tolist()
            stats = stats[stats["odds_bucket"].isin(valid_buckets)]
        else:
            stats = stats.iloc[0:0]

    # -------------------------------
    # KPI globali
    # -------------------------------
    if scala_type == "1X2":
        main_win = sub["is_home_win"].mean()
        roi_global = compute_roi(sub["is_home_win"], sub["avg_home_odds"])
        main_label = "Winrate segno 1"
    else:
        main_win = sub["is_away_win"].mean()
        roi_global = compute_roi(sub["is_away_win"], sub["avg_away_odds"])
        main_label = "Winrate segno 2"

    col1, col2, col3 = st.columns(3)
    col1.metric("Match Scala Pura filtrati", len(sub))
    col2.metric(main_label, f"{main_win*100:.1f}%")
    col3.metric("ROI segno principale", f"{roi_global*100:.2f}%")

    # -------------------------------
    # Filtro numero minimo match per bucket
    # -------------------------------
    min_n = st.slider(
        "Match minimi per bucket",
        10, 500, 50, step=10,
        key=f"min_n_{scala_type}"
    )
    stats_f = stats[stats["n_matches"] >= min_n]

    if stats_f.empty:
        st.warning("Nessun bucket supera la soglia minima.")
        return

    # -------------------------------
    # Grafico Winrate + ROI
    # -------------------------------
    st.markdown("### Andamento per quota favorita")
    chart_df = stats_f.set_index("bucket_mid")[["winrate_main_sign", "roi_main_sign"]]
    chart_df.columns = ["Winrate segno", "ROI segno"]
    st.line_chart(chart_df)

    # -------------------------------
    # Tabella buckets
    # -------------------------------
    st.markdown("### Tabella buckets")
    st.dataframe(stats_f)

    # -------------------------------
    # Dettaglio match per bucket
    # -------------------------------
    st.markdown("### Dettaglio match (per bucket)")

    buckets = stats_f["odds_bucket"].tolist()
    sel_bucket = st.selectbox("Seleziona bucket", buckets)

    sub_bucket = sub[sub["odds_bucket"] == sel_bucket].copy()

    st.write(f"Match nel bucket {sel_bucket}: {len(sub_bucket)}")

    # Coerce a bool per OR sicuri
    is1 = sub_bucket["is_home_win"].astype(bool)
    isx = sub_bucket["is_draw"].astype(bool)
    is2 = sub_bucket["is_away_win"].astype(bool)

    # Distribuzione 1/X/2
    colA, colB, colC = st.columns(3)
    colA.metric("P(1)", f"{is1.mean()*100:.1f}%")
    colB.metric("P(X)", f"{isx.mean()*100:.1f}%")
    colC.metric("P(2)", f"{is2.mean()*100:.1f}%")

    # doppie chance
    colD, colE, colF = st.columns(3)
    colD.metric("P(1X)", f"{((is1 | isx).mean())*100:.1f}%")
    colE.metric("P(X2)", f"{((isx | is2).mean())*100:.1f}%")
    colF.metric("P(12)", f"{((is1 | is2).mean())*100:.1f}%")

    # OU
    colG, colH = st.columns(2)
    colG.metric("Over 1.5", f"{sub_bucket['is_over15'].mean()*100:.1f}%")
    colH.metric("Over 2.5", f"{sub_bucket['is_over25'].mean()*100:.1f}%")

    # MG
    colI, colJ, colK = st.columns(3)
    colI.metric("MG Casa 1-3", f"{sub_bucket['mg_home_1_3'].mean()*100:.1f}%")
    colJ.metric("MG Casa 1-4", f"{sub_bucket['mg_home_1_4'].mean()*100:.1f}%")
    colK.metric("MG Casa 1-5", f"{sub_bucket['mg_home_1_5'].mean()*100:.1f}%")

    colL, colM, colN = st.columns(3)
    colL.metric("MG Ospite 1-3", f"{sub_bucket['mg_away_1_3'].mean()*100:.1f}%")
    colM.metric("MG Ospite 1-4", f"{sub_bucket['mg_away_1_4'].mean()*100:.1f}%")
    colN.metric("MG Ospite 1-5", f"{sub_bucket['mg_away_1_5'].mean()*100:.1f}%")

    # Tabella match
    st.dataframe(
        sub_bucket[
            [
                "date", "league", "home_team", "away_team",
                "avg_home_odds", "avg_draw_odds", "avg_away_odds",
                "pic_p1", "pic_px", "pic_p2",
                "home_ft", "away_ft"
            ]
        ]
    )

    st.download_button(
        "Scarica CSV bucket",
        data=sub_bucket.to_csv(index=False),
        file_name=f"scala_pura_{scala_type}_bucket_{sel_bucket}.csv",
        mime="text/csv"
    )
# -------------------------------------------------------------------
# MAIN STREAMLIT
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Scala Pura 1X2 / 2X1 – Explorer",
        layout="wide",
    )

    st.title("Scala Pura 1X2 / 2X1 – Dashboard")

    with st.sidebar:
        st.markdown("### Info")
        st.write(
            """
            Questa dashboard esplora la **Scala Pura**:

            - **Scala 1X2**:  
              pic_p1 > pic_px > pic_p2  
              bk_p1  > bk_px  > bk_p2

            - **Scala 2X1**:  
              pic_p2 > pic_px > pic_p1  
              bk_p2  > bk_px  > bk_p1

            I bucket sono per **quota favorita** con passo 0.10 (SMALL).
            """
        )

    df = load_datamaster()

    tab1, tab2 = st.tabs(["Scala Pura 1X2 (favorita casa)", "Scala Pura 2X1 (favorita ospite)"])

    with tab1:
        render_scala_tab(df, "1X2")

    with tab2:
        render_scala_tab(df, "2X1")


if __name__ == "__main__":
    main()
