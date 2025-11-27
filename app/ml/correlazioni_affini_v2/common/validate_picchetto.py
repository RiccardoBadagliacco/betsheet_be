import pandas as pd
import numpy as np

def get_shape(p1, px, p2):
    """Restituisce una stringa che descrive l'ordine delle probabilità."""
    tripla = [("1", p1), ("X", px), ("2", p2)]
    tripla = sorted(tripla, key=lambda x: x[1], reverse=True)
    return ">".join([t[0] for t in tripla])


def build_1x2_decision_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---------------------------------------------------------
    # Validazione minima colonne
    # ---------------------------------------------------------
    required = ["pic_p1", "pic_px", "pic_p2", "bk_p1", "bk_px", "bk_p2"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Manca colonna richiesta: {c}")

    # Rinominiamo per semplicità
    df["p1"] = df["pic_p1"]
    df["px"] = df["pic_px"]
    df["p2"] = df["pic_p2"]

    df["q1"] = df["bk_p1"]
    df["qX"] = df["bk_px"]
    df["q2"] = df["bk_p2"]

    # ---------------------------------------------------------
    # Favorito modello/book
    # ---------------------------------------------------------
    df["fav_mod"]  = df[["p1", "px", "p2"]].idxmax(axis=1)
    df["fav_book"] = df[["q1", "qX", "q2"]].idxmax(axis=1)

    # ---------------------------------------------------------
    # Gap modello/book (forza del favorito)
    # ---------------------------------------------------------
    def calc_gap(row, cols):
        vals = [row[c] for c in cols]
        sorted_vals = sorted(vals, reverse=True)
        return sorted_vals[0] - sorted_vals[1]

    df["gap_mod"]  = df.apply(lambda r: calc_gap(r, ["p1", "px", "p2"]), axis=1)
    df["gap_book"] = df.apply(lambda r: calc_gap(r, ["q1", "qX", "q2"]), axis=1)

    # ---------------------------------------------------------
    # Delta fra probabilità modello e bookmaker
    # ---------------------------------------------------------
    df["delta_p1"] = df["p1"] - df["q1"]
    df["delta_px"] = df["px"] - df["qX"]
    df["delta_p2"] = df["p2"] - df["q2"]

    # ---------------------------------------------------------
    # Shape delle probabilità (es. 1>X>2)
    # ---------------------------------------------------------
    df["shape_mod"]  = df.apply(lambda r: get_shape(r["p1"], r["px"], r["p2"]), axis=1)
    df["shape_book"] = df.apply(lambda r: get_shape(r["q1"], r["qX"], r["q2"]), axis=1)

    # ---------------------------------------------------------
    # Probabilità doppie chance (modello)
    # ---------------------------------------------------------
    df["p_1X"] = df["p1"] + df["px"]
    df["p_12"] = df["p1"] + df["p2"]
    df["p_X2"] = df["px"] + df["p2"]

    # ---------------------------------------------------------
    # Indicatori utili per le decisioni
    # ---------------------------------------------------------
    df["is_big_fav_mod"] = (df["p1"] >= 0.60) | (df["p2"] >= 0.60)
    df["is_no_draw"]     = df["px"] <= 0.20
    df["is_draw_match"]  = df["px"] >= 0.35
    df["is_mismatch_mod_book"] = df["fav_mod"] != df["fav_book"]

    return df


if __name__ == "__main__":
    print("📥 Carico dataset con picchetto e bookmaker...")
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent.parent  # .../correlazioni_affini_v2
    candidate_files = [
        base_dir / "data" / "step1_picchetto.parquet",
        Path.cwd() / "data" / "step1_picchetto.parquet",
        base_dir / "data" / "step2a_features_with_picchetto_fix.parquet",
        Path.cwd() / "data" / "step2a_features_with_picchetto_fix.parquet",
    ]
    dataset_path = next((p for p in candidate_files if p.exists()), None)
    if not dataset_path:
        raise FileNotFoundError(
            "Dataset non trovato. Cercati: "
            + ", ".join(str(p) for p in candidate_files)
        )

    df = pd.read_parquet(dataset_path)

    print("🔧 Generazione feature decisionali 1X2...")
    df_feat = build_1x2_decision_features(df)

    print("💾 Salvo dataset arricchito...")
    df_feat.to_parquet("data/dataset_1x2_decision_features.parquet", index=False)

    print("🎉 Fatto!")
