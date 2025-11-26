# app/ml/correlazioni_affini_v2/v2/scenario_profiles_v2.py

import sys
from pathlib import Path
from typing import Dict, Callable, Any

import numpy as np
import pandas as pd

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
                print(f"   ✔️  {name} trovato in: {candidate}")
                return candidate

    raise FileNotFoundError(
        f"File {names} non trovato. Cercato in {local_data_dir}, {common_data_dir}, {root_data_dir}"
    )


# -------------------------------------------------------------------
# Feature ausiliarie
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

    # delta_p* se mancanti
    if "delta_p1" not in df.columns:
        df["delta_p1"] = df["pic_p1"] - df["bk_p1"]
    if "delta_px" not in df.columns:
        df["delta_px"] = df["pic_px"] - df["bk_px"]
    if "delta_p2" not in df.columns:
        df["delta_p2"] = df["pic_p2"] - df["bk_p2"]

    # gap_mod se mancante
    if "gap_mod" not in df.columns:
        probs = df[["pic_p1", "pic_px", "pic_p2"]].astype(float)
        top2 = np.sort(probs.values, axis=1)[:, -2:]
        df["gap_mod"] = top2[:, 1] - top2[:, 0]

    # gap_book
    probs_bk = df[["bk_p1", "bk_px", "bk_p2"]].astype(float)
    top2_bk = np.sort(probs_bk.values, axis=1)[:, -2:]
    df["gap_book"] = top2_bk[:, 1] - top2_bk[:, 0]

    # entropy modello (1X2)
    def shannon_entropy(row):
        p = np.array([row["pic_p1"], row["pic_px"], row["pic_p2"]], dtype=float)
        p = np.clip(p, 1e-12, 1.0)
        p /= p.sum()
        return -np.sum(p * np.log(p))

    df["entropy_model_1x2"] = df.apply(shannon_entropy, axis=1)

    # forme MG
    df = add_mg_columns(df)

    return df


# -------------------------------------------------------------------
# Definizione scenari
# -------------------------------------------------------------------

def get_scenarios(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Restituisce un dict:
      { scenario_key: {"label": ..., "mask": callable(df)->SeriesBool} }
    soglie tarabili, ma già ragionevoli.
    """

    def s(name: str, label: str, mask_func: Callable[[pd.DataFrame], pd.Series]) -> Dict[str, Any]:
        return {"name": name, "label": label, "mask": mask_func}

    scenarios = {}

    # A) Modello molto più bullish sulla squadra di casa (1) rispetto al book
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

    # B) Modello molto più bullish sulla squadra ospite (2) rispetto al book
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

    # C) Mercato più aggressivo del modello sulla squadra di casa (1)
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

    # D) Mercato più aggressivo del modello sulla squadra ospite (2)
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

    # E) Favourito modello ≠ favorito mercato (1 vs 2)
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

    # F) Partite bilanciate, ma modello inclina 1
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


# -------------------------------------------------------------------
# Funzioni di aggregazione
# -------------------------------------------------------------------

def compute_global_baseline(df: pd.DataFrame) -> Dict[str, float]:
    """
    Distribuzione globale su tutto il dataset (per confronto).
    """
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


def compute_fav_stats(sub: pd.DataFrame) -> Dict[str, float]:
    """
    Statistiche su favorito del modello vs favorito del mercato
    (considerando solo favoriti 1 o 2, escludo X).
    """
    sub = sub.copy()

    # MODEL FAV
    mask_mod_home = sub["fav_mod"] == "pic_p1"
    mask_mod_away = sub["fav_mod"] == "pic_p2"

    model_fav_win = pd.Series(index=sub.index, dtype=float)
    model_fav_prob = pd.Series(index=sub.index, dtype=float)

    model_fav_win[mask_mod_home] = sub.loc[mask_mod_home, "is_home_win"]
    model_fav_prob[mask_mod_home] = sub.loc[mask_mod_home, "pic_p1"]

    model_fav_win[mask_mod_away] = sub.loc[mask_mod_away, "is_away_win"]
    model_fav_prob[mask_mod_away] = sub.loc[mask_mod_away, "pic_p2"]

    # BOOK FAV
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

    # delta main sign (solo dove modello e mercato hanno stesso main sign 1 o 2)
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


def compute_scenario_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le stats per ogni scenario definito.
    """
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

        # Base distribuzioni 1X2 + DC + OU + MG
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
# MAIN CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("📥 Carico datamaster...")
    dm_path = _resolve(["datamaster_1x2.parquet"])
    df = pd.read_parquet(dm_path)

    print("🔧 Aggiungo feature ausiliarie...")
    df = add_aux_features(df)

    print("🔧 Calcolo scenario profiles v2...")
    profiles = compute_scenario_profiles(df)

    out_path = _resolve(["."]) / "scenario_profiles_v2.csv"
    print(f"💾 Salvo scenario profiles in {out_path}")
    profiles.to_csv(out_path, index=False)

    print("\n📊 Anteprima:")
    print(profiles.head(20))
    print("\n🎉 Fatto!")