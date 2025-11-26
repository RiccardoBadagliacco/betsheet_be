# app/ml/correlazioni_affini_v2/v2/scala_pura_analysis.py

import sys
from pathlib import Path
from typing import List, Tuple, Literal, Dict, Any

import numpy as np
import pandas as pd

ScalaType = Literal["1X2", "2X1"]


# -------------------------------------------------------------------
# Utility per path (stile new_master_data)
# -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]

def _resolve(names: List[str]) -> Path:
    """
    Cerca il primo file esistente tra le directory note.
    """
    script_dir = Path(__file__).resolve().parent
    local_data_dir = script_dir / "data"
    common_data_dir = script_dir.parent / "data"
    root_data_dir = REPO_ROOT / "data"

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
# Scala Pura: definizioni
# -------------------------------------------------------------------
def add_scala_pura_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge:
      - is_scala_1x2
      - is_scala_2x1
      - scala_type: '1X2' | '2X1' | None
    Scala Pura 1X2:
        pic_p1 > pic_px > pic_p2  AND  bk_p1 > bk_px > bk_p2
    Scala Pura 2X1:
        pic_p2 > pic_px > pic_p1  AND  bk_p2 > bk_px > bk_p1
    """
    df = df.copy()

    required = ["pic_p1", "pic_px", "pic_p2", "bk_p1", "bk_px", "bk_p2"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"add_scala_pura_flags: manca colonna {c}")

    # NaN-safe: sostituisco temporaneamente con -1 per garantirci confronti falsi in caso di NaN
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

    # scala_type string
    df["scala_type"] = None
    df.loc[is_scala_1x2, "scala_type"] = "1X2"
    df.loc[is_scala_2x1, "scala_type"] = "2X1"

    return df


# -------------------------------------------------------------------
# Goal-range (MG Casa / Ospite)
# -------------------------------------------------------------------
def add_mg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge:
      - mg_home_1_3, mg_home_1_4, mg_home_1_5
      - mg_away_1_3, mg_away_1_4, mg_away_1_5
    """
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


# -------------------------------------------------------------------
# Bucket odds SMALL [1.01, 2.50) con step 0.10
# -------------------------------------------------------------------
def build_odds_buckets(
    fav_odds: pd.Series,
    low: float = 1.01,
    high: float = 2.50,
    step: float = 0.10,
) -> Tuple[pd.Series, List[str], np.ndarray]:
    """
    Restituisce:
      - series con label bucket
      - lista labels
      - array edges
    """
    edges = np.arange(low, high + step, step)
    labels = []
    for i in range(len(edges) - 1):
        l = edges[i]
        r = edges[i + 1]
        labels.append(f"{l:.2f}-{r:.2f}")

    bucket = pd.cut(fav_odds, bins=edges, labels=labels, right=False, include_lowest=True)
    return bucket, labels, edges


# -------------------------------------------------------------------
# ROI helper
# -------------------------------------------------------------------
def compute_roi(outcome: pd.Series, odds: pd.Series) -> float:
    """
    Flat stake=1, ROI = (sum(win*odds) - N) / N
    outcome: bool/0-1
    """
    mask = outcome.astype(bool)
    n = len(outcome)
    if n == 0:
        return np.nan
    winnings = odds[mask].fillna(0).sum()
    roi = (winnings - n) / n
    return float(roi)


# -------------------------------------------------------------------
# Analisi per Scala e bucket
# -------------------------------------------------------------------
def analyze_scala(df: pd.DataFrame, scala_type: ScalaType) -> pd.DataFrame:
    """
    df: datamaster con:
      - pic_p1, pic_px, pic_p2, bk_p1, bk_px, bk_p2
      - avg_home_odds, avg_away_odds
      - is_home_win, is_draw, is_away_win
      - is_over15, is_over25, is_under25
      - mg_home_1_3, mg_home_1_4, mg_home_1_5, mg_away_1_3, mg_away_1_4, mg_away_1_5
      - is_scala_1x2, is_scala_2x1
    """
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
    sub["fav_odds"] = fav_odds[mask]

    # costruiamo bucket
    sub["odds_bucket"], labels, edges = build_odds_buckets(sub["fav_odds"])

    results = []
    for label in labels:
        g = sub[sub["odds_bucket"] == label]
        n = len(g)
        if n == 0:
            continue

        # distribuzione 1/X/2
        p1 = g["is_home_win"].mean()
        pX = g["is_draw"].mean()
        p2 = g["is_away_win"].mean()

        # doppie chance
        freq_1X = ((g["is_home_win"].astype(bool)) | (g["is_draw"].astype(bool))).mean()
        freq_X2 = ((g["is_draw"].astype(bool)) | (g["is_away_win"].astype(bool))).mean()
        freq_12 = ((g["is_home_win"].astype(bool)) | (g["is_away_win"].astype(bool))).mean()

        # ROI segno coerente con la scala
        if scala_type == "1X2":
            odds_main = g["avg_home_odds"].astype(float)
        else:
            odds_main = g["avg_away_odds"].astype(float)
        roi_main = compute_roi(outcome_main[g.index], odds_main)

        # Over/Under 1.5 e 2.5
        p_over15 = g["is_over15"].mean() if "is_over15" in g.columns else np.nan
        p_under15 = 1.0 - p_over15 if p_over15 == p_over15 else np.nan  # NaN-safe

        p_over25 = g["is_over25"].mean() if "is_over25" in g.columns else np.nan
        if "is_under25" in g.columns:
            p_under25 = g["is_under25"].mean()
        else:
            p_under25 = 1.0 - p_over25 if p_over25 == p_over25 else np.nan

        # MG Casa / Ospite
        p_home_1_3 = g["mg_home_1_3"].mean()
        p_home_1_4 = g["mg_home_1_4"].mean()
        p_home_1_5 = g["mg_home_1_5"].mean()

        p_away_1_3 = g["mg_away_1_3"].mean()
        p_away_1_4 = g["mg_away_1_4"].mean()
        p_away_1_5 = g["mg_away_1_5"].mean()

        # bucket mid
        l_float = float(label.split("-")[0])
        r_float = float(label.split("-")[1])
        bucket_mid = 0.5 * (l_float + r_float)

        results.append(
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

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values("bucket_mid")
    return res_df


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    script_dir = Path(__file__).resolve().parent
    local_data_dir = script_dir / "data"
    local_data_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Carico datamaster...")
    dm_path = _resolve(["datamaster_1x2.parquet"])
    df = pd.read_parquet(dm_path)

    print("🔧 Aggiungo flag Scala Pura...")
    df = add_scala_pura_flags(df)

    print("🔧 Aggiungo colonne MG casa/ospite...")
    df = add_mg_columns(df)

    print("📊 Analisi Scala Pura 1X2...")
    res_1x2 = analyze_scala(df, "1X2")

    print("📊 Analisi Scala Pura 2X1...")
    res_2x1 = analyze_scala(df, "2X1")

    out_1 = local_data_dir / "scala_pura_1x2_small.csv"
    out_2 = local_data_dir / "scala_pura_2x1_small.csv"

    print(f"💾 Salvo {out_1}...")
    res_1x2.to_csv(out_1, index=False)

    print(f"💾 Salvo {out_2}...")
    res_2x1.to_csv(out_2, index=False)

    print("🎉 Analisi Scala Pura completata!")


if __name__ == "__main__":
    main()