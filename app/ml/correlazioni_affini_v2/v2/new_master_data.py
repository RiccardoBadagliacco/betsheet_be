import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.correlazioni_affini_v2.common.picchetto import apply_picchetto_tech_fix


# ============================================================
# Aiuto per shape della tripla probabilistica
# ============================================================
def get_shape(p1, px, p2):
    arr = [("1", p1), ("X", px), ("2", p2)]
    arr = sorted(arr, key=lambda x: x[1], reverse=True)
    return ">".join([a[0] for a in arr])


# ============================================================
# Build final DATA MASTER 1X2
# ============================================================
def build_datamaster(df_step1c):

    df = df_step1c.copy()

    # --------------------------------------------------------
    # RAW DIFFS — ELO (PRE ONLY) + FORM
    # --------------------------------------------------------
    df["elo_diff_raw"] = df["elo_home_pre"] - df["elo_away_pre"]
    df["form_pts_diff_raw"] = df["home_form_pts_avg_lastN"] - df["away_form_pts_avg_lastN"]
    df["form_gf_diff_raw"] = df["home_form_gf_avg_lastN"] - df["away_form_gf_avg_lastN"]
    df["form_ga_diff_raw"] = df["away_form_ga_avg_lastN"] - df["home_form_ga_avg_lastN"]

    # --------------------------------------------------------
    # 1. Applica picchetto tecnico 1X2
    # --------------------------------------------------------
    print("🔧 Calcolo Picchetto Tecnico 1X2...")
    df_pic = apply_picchetto_tech_fix(df.copy(), alpha=0.3, stats=None)

    df["pic_p1"] = df_pic["pic_p1"]
    df["pic_px"] = df_pic["pic_px"]
    df["pic_p2"] = df_pic["pic_p2"]

    # Quote picchetto
    df["pic_odd1"] = 1 / df["pic_p1"]
    df["pic_oddX"] = 1 / df["pic_px"]
    df["pic_odd2"] = 1 / df["pic_p2"]

    # --------------------------------------------------------
    # 2. Delta probabilità (modello - bookmaker)
    # --------------------------------------------------------
    df["delta_p1"] = df["pic_p1"] - df["bk_p1"]
    df["delta_px"] = df["pic_px"] - df["bk_px"]
    df["delta_p2"] = df["pic_p2"] - df["bk_p2"]

    # --------------------------------------------------------
    # 3. Delta quote (picchetto vs book)
    # --------------------------------------------------------
    df["delta_q1"] = df["pic_odd1"] - df["avg_home_odds"]
    df["delta_qX"] = df["pic_oddX"] - df["avg_draw_odds"]
    df["delta_q2"] = df["pic_odd2"] - df["avg_away_odds"]

    # --------------------------------------------------------
    # 4. Shape e favoriti
    # --------------------------------------------------------
    df["shape_mod"] = df.apply(lambda r: get_shape(r["pic_p1"], r["pic_px"], r["pic_p2"]), axis=1)
    df["shape_book"] = df.apply(lambda r: get_shape(r["bk_p1"], r["bk_px"], r["bk_p2"]), axis=1)

    df["fav_mod"] = df[["pic_p1", "pic_px", "pic_p2"]].idxmax(axis=1)
    df["fav_book"] = df[["bk_p1", "bk_px", "bk_p2"]].idxmax(axis=1)

    df["mismatch_mod_book"] = df["fav_mod"] != df["fav_book"]

    # gap modello (forza del segno favorito)
    def calc_gap(row):
        arr = sorted([row["pic_p1"], row["pic_px"], row["pic_p2"]], reverse=True)
        return arr[0] - arr[1]

    df["gap_mod"] = df.apply(calc_gap, axis=1)

    # --------------------------------------------------------
    # 5. Probabilità doppie chance (modello)
    # --------------------------------------------------------
    df["p_1X"] = df["pic_p1"] + df["pic_px"]
    df["p_X2"] = df["pic_px"] + df["pic_p2"]
    df["p_12"] = df["pic_p1"] + df["pic_p2"]

    from app.ml.correlazioni_affini_v2.common.picchetto_v2 import (
        fit_picchetto_v2_stats,
        apply_picchetto_tech_v2
    )

    print("🔧 Calcolo Picchetto Tecnico V2...")
    stats_v2 = fit_picchetto_v2_stats(df)
    df_v2 = apply_picchetto_tech_v2(df, stats=stats_v2)

    df["pic_v2_p1"] = df_v2["pic_v2_p1"]
    df["pic_v2_px"] = df_v2["pic_v2_px"]
    df["pic_v2_p2"] = df_v2["pic_v2_p2"]

    df["pic_v2_odd1"] = df_v2["pic_v2_odd1"]
    df["pic_v2_oddX"] = df_v2["pic_v2_oddX"]
    df["pic_v2_odd2"] = df_v2["pic_v2_odd2"]

    # Calcola y_1x2 e result_str
    df["y_1x2"] = df.apply(lambda r: 1 if r["home_ft"] > r["away_ft"] else (2 if r["home_ft"] < r["away_ft"] else 0), axis=1)
    df["result_str"] = df["y_1x2"].map({1: "1", 0: "X", 2: "2"})

    print("🎉 Datamaster 1X2 pronto!")
    return df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    LOCAL_DATA_DIR = SCRIPT_DIR / "data"           # dati/output vicino allo script
    COMMON_DATA_DIR = SCRIPT_DIR.parent / "data"    # data pipeline affini
    ROOT_DATA_DIR = REPO_ROOT / "data"              # fallback root
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _resolve(names):
        """
        Cerca il primo file esistente tra le directory note.
        """
        if isinstance(names, (str, Path)):
            names = [names]
        for name in names:
            for base in (LOCAL_DATA_DIR, COMMON_DATA_DIR, ROOT_DATA_DIR):
                candidate = base / name
                if candidate.exists():
                    print(f"   ✔️  {name} trovato in: {candidate}")
                    return candidate
        raise FileNotFoundError(
            f"File {names} non trovato. Cercato in {LOCAL_DATA_DIR}, {COMMON_DATA_DIR}, {ROOT_DATA_DIR}"
        )

    STEP1C_PATH = _resolve(["step1c_dataset_with_elo_form.parquet", "step1c_elo_form_dataset.parquet"])
    OUT_PATH = LOCAL_DATA_DIR / "datamaster_1x2.parquet"

    print("📥 Carico step1c dataset...")
    df_step1c = pd.read_parquet(STEP1C_PATH)

    df_master = build_datamaster(df_step1c)

    print("💾 Salvo datamaster...")
    df_master.to_parquet(OUT_PATH, index=False)

    print(f"🎯 DATAMASTER SALVATO CON SUCCESSO in {OUT_PATH}!")
