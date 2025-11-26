# app/ml/correlazioni_affini_v2/v2/scenario_profiles_v2_dashboard.py

import sys
from pathlib import Path
from typing import Dict, Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -------------------------------------------------------------------
# Path helper
# -------------------------------------------------------------------
def _resolve(names):
    script_dir = Path(__file__).resolve().parent
    local_data_dir = script_dir / "data"
    common_data_dir = script_dir.parent / "data"
    root_data_dir = REPO_ROOT / "data"

    if isinstance(names, (str, Path)):
        names = [names]

    for name in names:
        for base in (local_data_dir, common_data_dir, root_data_dir):
            candidate = base / name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"File {names} non trovato. Cercato in {local_data_dir}, {common_data_dir}, {root_data_dir}"
    )


# -------------------------------------------------------------------
# Feature ausiliarie (copiate da scenario_profiles_v2)
# -------------------------------------------------------------------
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


def add_aux_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "delta_p1" not in df.columns:
        df["delta_p1"] = df["pic_p1"] - df["bk_p1"]
    if "delta_px" not in df.columns:
        df["delta_px"] = df["pic_px"] - df["bk_px"]
    if "delta_p2" not in df.columns:
        df["delta_p2"] = df["pic_p2"] - df["bk_p2"]

    if "gap_mod" not in df.columns:
        probs = df[["pic_p1", "pic_px", "pic_p2"]].astype(float)
        top2 = np.sort(probs.values, axis=1)[:, -2:]
        df["gap_mod"] = top2[:, 1] - top2[:, 0]

    probs_bk = df[["bk_p1", "bk_px", "bk_p2"]].astype(float)
    top2_bk = np.sort(probs_bk.values, axis=1)[:, -2:]
    df["gap_book"] = top2_bk[:, 1] - top2_bk[:, 0]

    def shannon_entropy(row):
        p = np.array([row["pic_p1"], row["pic_px"], row["pic_p2"]], dtype=float)
        p = np.clip(p, 1e-12, 1.0)
        p /= p.sum()
        return -np.sum(p * np.log(p))

    df["entropy_model_1x2"] = df.apply(shannon_entropy, axis=1)

    df = add_mg_columns(df)

    return df


# -------------------------------------------------------------------
# Scenari (copiati da scenario_profiles_v2)
# -------------------------------------------------------------------
def get_scenarios(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def s(name: str, label: str, mask_func: Callable[[pd.DataFrame], pd.Series]) -> Dict[str, Any]:
        return {"name": name, "label": label, "mask": mask_func}

    scenarios = {}

    scenarios["A_model_strong_home_vs_market"] = s(
        "A_model_strong_home_vs_market",
        "A • Modello forte su 1, mercato più prudente",
        lambda d: (
            (d["fav_mod"] == "pic_p1") &
            (d["fav_book"] == "bk_p1") &
            (d["delta_p1"] >= 0.06) &
            (d["gap_mod"] >= 0.18) &
            (d["entropy_bk_1x2"] <= 1.05)
        )
    )

    scenarios["B_model_strong_away_vs_market"] = s(
        "B_model_strong_away_vs_market",
        "B • Modello forte su 2, mercato più prudente",
        lambda d: (
            (d["fav_mod"] == "pic_p2") &
            (d["fav_book"] == "bk_p2") &
            (d["delta_p2"] >= 0.06) &
            (d["gap_mod"] >= 0.18) &
            (d["entropy_bk_1x2"] <= 1.05)
        )
    )

    scenarios["C_market_strong_home_vs_model"] = s(
        "C_market_strong_home_vs_model",
        "C • Mercato forte su 1, modello più prudente",
        lambda d: (
            (d["fav_mod"] == "pic_p1") &
            (d["fav_book"] == "bk_p1") &
            (d["delta_p1"] <= -0.05) &
            (d["gap_book"] >= 0.18)
        )
    )

    scenarios["D_market_strong_away_vs_model"] = s(
        "D_market_strong_away_vs_model",
        "D • Mercato forte su 2, modello più prudente",
        lambda d: (
            (d["fav_mod"] == "pic_p2") &
            (d["fav_book"] == "bk_p2") &
            (d["delta_p2"] <= -0.05) &
            (d["gap_book"] >= 0.18)
        )
    )

    scenarios["E_opposite_favourites_1_vs_2"] = s(
        "E_opposite_favourites_1_vs_2",
        "E • Modello e mercato in disaccordo (1 vs 2)",
        lambda d: (
            (d["mismatch_mod_book"]) &
            (d["fav_mod"].isin(["pic_p1", "pic_p2"])) &
            (d["fav_book"].isin(["bk_p1", "bk_p2"])) &
            (np.abs(d["delta_p1"]) >= 0.03) &
            (np.abs(d["delta_p2"]) >= 0.03)
        )
    )

    scenarios["F_balanced_but_model_prefers_home"] = s(
        "F_balanced_but_model_prefers_home",
        "F • Quote bilanciate, modello preferisce 1",
        lambda d: (
            (np.abs(d["pic_p1"] - d["pic_p2"]) <= 0.04) &
            (d["fav_mod"] == "pic_p1") &
            (d["fav_book"] != "bk_p1") &
            (d["entropy_model_1x2"] >= 1.05)
        )
    )

    return scenarios


def compute_fav_stats(sub: pd.DataFrame) -> Dict[str, float]:
    sub = sub.copy()

    mask_mod_home = sub["fav_mod"] == "pic_p1"
    mask_mod_away = sub["fav_mod"] == "pic_p2"

    model_fav_win = pd.Series(index=sub.index, dtype=float)
    model_fav_prob = pd.Series(index=sub.index, dtype=float)

    model_fav_win[mask_mod_home] = sub.loc[mask_mod_home, "is_home_win"]
    model_fav_prob[mask_mod_home] = sub.loc[mask_mod_home, "pic_p1"]

    model_fav_win[mask_mod_away] = sub.loc[mask_mod_away, "is_away_win"]
    model_fav_prob[mask_mod_away] = sub.loc[mask_mod_away, "pic_p2"]

    mask_bk_home = sub["fav_book"] == "bk_p1"
    mask_bk_away = sub["fav_book"] == "bk_p2"

    market_fav_win = pd.Series(index=sub.index, dtype=float)
    market_fav_prob = pd.Series(index=sub.index, dtype=float)
    market_fav_odds = pd.Series(index=sub.index, dtype=float)

    market_fav_win[mask_bk_home] = sub.loc[mask_bk_home, "is_home_win"]
    market_fav_prob[mask_bk_home] = sub.loc[mask_bk_home, "bk_p1"]
    market_fav_odds[mask_bk_home] = sub.loc[mask_bk_home, "avg_home_odds"]

    market_fav_win[mask_bk_away] = sub.loc[mask_bk_away, "is_away_win"]
    market_fav_prob[mask_bk_away] = sub.loc[mask_bk_away, "bk_p2"]
    market_fav_odds[mask_bk_away] = sub.loc[mask_bk_away, "avg_away_odds"]

    aligned_home = mask_mod_home & mask_bk_home
    aligned_away = mask_mod_away & mask_bk_away
    aligned = aligned_home | aligned_away

    delta_main = pd.Series(index=sub.index, dtype=float)
    delta_main[aligned_home] = sub.loc[aligned_home, "delta_p1"]
    delta_main[aligned_away] = sub.loc[aligned_away, "delta_p2"]

    return {
        "model_fav_winrate": float(model_fav_win.mean(skipna=True)),
        "model_fav_prob": float(model_fav_prob.mean(skipna=True)),
        "market_fav_winrate": float(market_fav_win.mean(skipna=True)),
        "market_fav_prob": float(market_fav_prob.mean(skipna=True)),
        "market_fav_odds": float(market_fav_odds.mean(skipna=True)),
        "delta_main_avg": float(delta_main.mean(skipna=True)),
    }


def compute_global_baseline(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    out["matches"] = len(df)
    out["p1"] = df["is_home_win"].mean()
    out["pX"] = df["is_draw"].mean()
    out["p2"] = df["is_away_win"].mean()

    is1 = df["is_home_win"].astype(bool)
    isx = df["is_draw"].astype(bool)
    is2 = df["is_away_win"].astype(bool)

    out["p_1X"] = (is1 | isx).mean()
    out["p_X2"] = (isx | is2).mean()
    out["p_12"] = (is1 | is2).mean()

    out["over15"] = df["is_over15"].mean() if "is_over15" in df.columns else np.nan
    out["over25"] = df["is_over25"].mean() if "is_over25" in df.columns else np.nan

    return out


def compute_scenario_profiles(df: pd.DataFrame) -> pd.DataFrame:
    scenarios = get_scenarios(df)
    rows = []

    for key, meta in scenarios.items():
        name = meta["name"]
        label = meta["label"]
        mask = meta["mask"](df)
        sub = df[mask].copy()
        n = len(sub)
        if n == 0:
            continue

        p1 = sub["is_home_win"].mean()
        pX = sub["is_draw"].mean()
        p2 = sub["is_away_win"].mean()

        is1 = sub["is_home_win"].astype(bool)
        isx = sub["is_draw"].astype(bool)
        is2 = sub["is_away_win"].astype(bool)

        p_1X = (is1 | isx).mean()
        p_X2 = (isx | is2).mean()
        p_12 = (is1 | is2).mean()

        over15 = sub["is_over15"].mean() if "is_over15" in sub.columns else np.nan
        over25 = sub["is_over25"].mean() if "is_over25" in sub.columns else np.nan

        mg_home_1_3 = sub["mg_home_1_3"].mean()
        mg_home_1_4 = sub["mg_home_1_4"].mean()
        mg_home_1_5 = sub["mg_home_1_5"].mean()

        mg_away_1_3 = sub["mg_away_1_3"].mean()
        mg_away_1_4 = sub["mg_away_1_4"].mean()
        mg_away_1_5 = sub["mg_away_1_5"].mean()

        fav_stats = compute_fav_stats(sub)

        row = {
            "scenario_key": key,
            "scenario_label": label,
            "matches": n,
            "p1": p1,
            "pX": pX,
            "p2": p2,
            "p_1X": p_1X,
            "p_X2": p_X2,
            "p_12": p_12,
            "over15": over15,
            "over25": over25,
            "mg_home_1_3": mg_home_1_3,
            "mg_home_1_4": mg_home_1_4,
            "mg_home_1_5": mg_home_1_5,
            "mg_away_1_3": mg_away_1_3,
            "mg_away_1_4": mg_away_1_4,
            "mg_away_1_5": mg_away_1_5,
            "model_fav_winrate": fav_stats["model_fav_winrate"],
            "model_fav_prob": fav_stats["model_fav_prob"],
            "market_fav_winrate": fav_stats["market_fav_winrate"],
            "market_fav_prob": fav_stats["market_fav_prob"],
            "market_fav_odds": fav_stats["market_fav_odds"],
            "delta_main_avg": fav_stats["delta_main_avg"],
        }
        rows.append(row)

    prof = pd.DataFrame(rows)
    if not prof.empty:
        prof = prof.sort_values("matches", ascending=False).reset_index(drop=True)
    return prof


# -------------------------------------------------------------------
# Cache data
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_datamaster() -> pd.DataFrame:
    dm_path = _resolve("datamaster_1x2.parquet")
    df = pd.read_parquet(dm_path)
    df = add_aux_features(df)
    return df


@st.cache_data(show_spinner=True)
def load_profiles(df: pd.DataFrame) -> pd.DataFrame:
    return compute_scenario_profiles(df)


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Scenario Profiles v2 – Model vs Market",
        layout="wide",
    )

    st.title("Scenario Profiles v2 – Modello vs Bookmaker")

    df = load_datamaster()
    baseline = compute_global_baseline(df)
    profiles = load_profiles(df)

    with st.sidebar:
        st.markdown("### Info")
        st.write(
            """
            Questa dashboard confronta **picchetto tecnico** e **quote di mercato**
            usando scenari strutturati:

            - Modello molto più bullish del book (1 o 2)
            - Mercato più aggressivo del modello
            - Favoriti in disaccordo (1 vs 2)
            - Partite bilanciate ma modello inclinato su 1

            Per ogni scenario vedi:
            - Distribuzione 1 / X / 2 e doppie chance
            - Over/Under 1.5 e 2.5
            - MG Casa / Ospite 1–3 / 1–4 / 1–5
            - Come si comportano **favorito del modello** vs **favorito del book**
            """
        )
        st.markdown("---")
        st.write(f"Match totali nel datamaster: **{baseline['matches']:,}**")

    # ----------------- Baseline globale -----------------
    st.subheader("Baseline globale (su tutto il datamaster)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(1)", f"{baseline['p1']*100:.1f} %")
    c2.metric("P(X)", f"{baseline['pX']*100:.1f} %")
    c3.metric("P(2)", f"{baseline['p2']*100:.1f} %")
    c4.metric("Match", f"{baseline['matches']:,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("P(1X)", f"{baseline['p_1X']*100:.1f} %")
    c6.metric("P(X2)", f"{baseline['p_X2']*100:.1f} %")
    c7.metric("P(12)", f"{baseline['p_12']*100:.1f} %")
    if not np.isnan(baseline["over25"]):
        c8.metric("Over 2.5", f"{baseline['over25']*100:.1f} %")

    st.markdown("---")

    # ----------------- Tabella scenari -----------------
    st.subheader("Panoramica scenari")
    if profiles.empty:
        st.warning("Nessuno scenario ha match sufficienti.")
        return

    show_cols = [
        "scenario_label",
        "matches",
        "p1",
        "pX",
        "p2",
        "p_1X",
        "p_X2",
        "p_12",
        "over25",
        "model_fav_winrate",
        "model_fav_prob",
        "market_fav_winrate",
        "market_fav_prob",
        "delta_main_avg",
    ]
    show_cols = [c for c in show_cols if c in profiles.columns]

    st.dataframe(
        profiles[show_cols].style.format(
            {
                "p1": "{:.3f}",
                "pX": "{:.3f}",
                "p2": "{:.3f}",
                "p_1X": "{:.3f}",
                "p_X2": "{:.3f}",
                "p_12": "{:.3f}",
                "over25": "{:.3f}",
                "model_fav_winrate": "{:.3f}",
                "model_fav_prob": "{:.3f}",
                "market_fav_winrate": "{:.3f}",
                "market_fav_prob": "{:.3f}",
                "delta_main_avg": "{:.3f}",
            }
        ),
        use_container_width=True,
        height=360,
    )

    st.markdown("---")

    # ----------------- Selettore scenario -----------------
    st.subheader("Dettaglio scenario")

    options = {row["scenario_label"]: row["scenario_key"] for _, row in profiles.iterrows()}
    label_selected = st.selectbox("Seleziona scenario", list(options.keys()))
    scenario_key = options[label_selected]

    meta = get_scenarios(df)[scenario_key]
    mask = meta["mask"](df)
    sub = df[mask].copy()

    if sub.empty:
        st.warning("Nessun match per questo scenario.")
        return

    st.markdown(f"**Scenario selezionato:** {meta['label']}")
    st.write(f"Match nello scenario: **{len(sub):,}**")

    # KPI 1X2
    is1 = sub["is_home_win"].astype(bool)
    isx = sub["is_draw"].astype(bool)
    is2 = sub["is_away_win"].astype(bool)

    p1 = is1.mean()
    pX = isx.mean()
    p2 = is2.mean()

    p_1X = (is1 | isx).mean()
    p_X2 = (isx | is2).mean()
    p_12 = (is1 | is2).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(1) scenario", f"{p1*100:.1f} %", delta=f"{(p1 - baseline['p1'])*100:.1f} pp")
    c2.metric("P(X) scenario", f"{pX*100:.1f} %", delta=f"{(pX - baseline['pX'])*100:.1f} pp")
    c3.metric("P(2) scenario", f"{p2*100:.1f} %", delta=f"{(p2 - baseline['p2'])*100:.1f} pp")
    c4.metric("Match scenario", f"{len(sub):,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("P(1X) scenario", f"{p_1X*100:.1f} %", delta=f"{(p_1X - baseline['p_1X'])*100:.1f} pp")
    c6.metric("P(X2) scenario", f"{p_X2*100:.1f} %", delta=f"{(p_X2 - baseline['p_X2'])*100:.1f} pp")
    c7.metric("P(12) scenario", f"{p_12*100:.1f} %", delta=f"{(p_12 - baseline['p_12'])*100:.1f} pp")

    if "is_over25" in sub.columns:
        over25_s = sub["is_over25"].mean()
        c8.metric("Over 2.5 scenario", f"{over25_s*100:.1f} %",
                  delta=f"{(over25_s - baseline['over25'])*100:.1f} pp")

    st.markdown("### Modello vs Mercato sul favorito")

    fav_stats = compute_fav_stats(sub)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Winrate favorito MODELLO",
        f"{fav_stats['model_fav_winrate']*100:.1f} %",
    )
    m2.metric(
        "Prob. media favorito MODELLO",
        f"{fav_stats['model_fav_prob']*100:.1f} %",
    )
    m3.metric(
        "Winrate favorito MERCATO",
        f"{fav_stats['market_fav_winrate']*100:.1f} %",
    )
    m4.metric(
        "Prob. media favorito MERCATO",
        f"{fav_stats['market_fav_prob']*100:.1f} %",
    )

    m5, m6 = st.columns(2)
    m5.metric("Quota media favorito MERCATO", f"{fav_stats['market_fav_odds']:.3f}")
    m6.metric("Δ prob. main (model - market)", f"{fav_stats['delta_main_avg']*100:.2f} pp")

    st.markdown("### Over / MG Casa / MG Ospite")

    col_g, col_h = st.columns(2)
    if "is_over15" in sub.columns:
        col_g.metric("Over 1.5", f"{sub['is_over15'].mean()*100:.1f} %")
    if "is_over25" in sub.columns:
        col_h.metric("Over 2.5", f"{sub['is_over25'].mean()*100:.1f} %")

    col_i, col_j, col_k = st.columns(3)
    col_i.metric("MG Casa 1-3", f"{sub['mg_home_1_3'].mean()*100:.1f} %")
    col_j.metric("MG Casa 1-4", f"{sub['mg_home_1_4'].mean()*100:.1f} %")
    col_k.metric("MG Casa 1-5", f"{sub['mg_home_1_5'].mean()*100:.1f} %")

    col_l, col_m, col_n = st.columns(3)
    col_l.metric("MG Ospite 1-3", f"{sub['mg_away_1_3'].mean()*100:.1f} %")
    col_m.metric("MG Ospite 1-4", f"{sub['mg_away_1_4'].mean()*100:.1f} %")
    col_n.metric("MG Ospite 1-5", f"{sub['mg_away_1_5'].mean()*100:.1f} %")

    st.markdown("### Sample match dello scenario")

    show_cols = [
        "date",
        "league",
        "home_team",
        "away_team",
        "avg_home_odds",
        "avg_draw_odds",
        "avg_away_odds",
        "pic_p1",
        "pic_px",
        "pic_p2",
        "home_ft",
        "away_ft",
    ]
    show_cols = [c for c in show_cols if c in sub.columns]

    st.dataframe(sub[show_cols].head(300), use_container_width=True)

    csv = sub.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Scarica CSV scenario",
        data=csv,
        file_name=f"scenario_{scenario_key}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()