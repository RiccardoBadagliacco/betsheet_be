# ============================================================
# STEP0C — VALIDAZIONE STEP0 + STEP0B (FORMA)
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
STEP0_PATH = DATA_DIR / "step0_profeta.parquet"


def main():
    print("📥 Carico STEP0:", STEP0_PATH)
    df = pd.read_parquet(STEP0_PATH)

    print("\n==============================")
    print("🔎 CHECK 1 — STRUTTURA")
    print("==============================")

    expected_form_cols = [
        "gf_last5", "ga_last5", "pts_last5", "gd_last5",
        "gf_last10", "ga_last10", "pts_last10", "gd_last10",
        "avg_gf_last5", "avg_ga_last5",
    ]

    for side in ["home", "away"]:
        for c in expected_form_cols:
            col = f"{c}_{side}"
            assert col in df.columns, f"❌ Manca colonna: {col}"

    bad_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    assert len(bad_cols) == 0, f"❌ Colonne spurie trovate: {bad_cols}"

    print("✅ Colonne OK")

    print("\n==============================")
    print("🔎 CHECK 2 — CARDINALITÀ")
    print("==============================")

    assert df["match_id"].is_unique, "❌ match_id duplicati"
    print(f"✅ {len(df)} match unici")

    print("\n==============================")
    print("🔎 CHECK 3 — VALORI IMPOSSIBILI")
    print("==============================")

    def assert_nonneg(series: pd.Series, name: str):
        s = pd.to_numeric(series, errors="coerce")  # forza numerico, invalido -> NaN
        neg = s.dropna() < 0
        nneg = int(neg.sum())
        if nneg > 0:
            print(f"❌ {name}: trovati {nneg} valori negativi. Esempi:")
            print(df.loc[neg.index[neg], ["match_id", name]].head(10))
            raise AssertionError(f"{name} contiene valori negativi")
        print(f"✅ {name}: ok (min={s.dropna().min() if s.notna().any() else 'NA'})")

    for side in ["home", "away"]:
        for c in ["gf_last5", "ga_last5", "pts_last5", "gf_last10", "ga_last10", "pts_last10"]:
            assert_nonneg(df[f"{c}_{side}"], f"{c}_{side}")

    # range punti (non-na)
    for side in ["home", "away"]:
        s5 = pd.to_numeric(df[f"pts_last5_{side}"], errors="coerce")
        s10 = pd.to_numeric(df[f"pts_last10_{side}"], errors="coerce")
        assert (s5.dropna() <= 15).all(), f"❌ pts_last5_{side} > 15"
        assert (s10.dropna() <= 30).all(), f"❌ pts_last10_{side} > 30"

    print("✅ Nessun valore impossibile")

    print("\n==============================")
    print("🔎 CHECK 4 — COERENZA MATEMATICA")
    print("==============================")

    for side in ["home", "away"]:
        gd5 = df[f"gf_last5_{side}"] - df[f"ga_last5_{side}"]
        gd10 = df[f"gf_last10_{side}"] - df[f"ga_last10_{side}"]

        assert np.allclose(
            df[f"gd_last5_{side}"].fillna(0),
            gd5.fillna(0)
        ), f"❌ gd_last5_{side} incoerente"

        assert np.allclose(
            df[f"gd_last10_{side}"].fillna(0),
            gd10.fillna(0)
        ), f"❌ gd_last10_{side} incoerente"

        assert np.allclose(
            df[f"avg_gf_last5_{side}"].fillna(0),
            (df[f"gf_last5_{side}"] / 5).fillna(0)
        ), f"❌ avg_gf_last5_{side} incoerente"

    print("✅ Coerenza matematica OK")

    print("\n==============================")
    print("🔎 CHECK 5 — DATA LEAKAGE (EARLY MATCHES)")
    print("==============================")

    early_nan = df[
        (df["gf_last5_home"].isna()) &
        (df["gf_last5_away"].isna())
    ]

    print(f"ℹ️ Match senza forma (inizio stagione): {len(early_nan)}")
    assert len(early_nan) > 0, "❌ Nessun match senza forma → possibile leakage"

    print("✅ Nessun data leakage evidente")

    print("\n==============================")
    print("🔎 CHECK 6 — DISTRIBUZIONI")
    print("==============================")

    for col in [
        "gf_last5_home", "ga_last5_home",
        "gf_last10_home", "ga_last10_home",
        "pts_last5_home", "pts_last10_home"
    ]:
        print(f"\n📊 {col}")
        print(df[col].describe())

    print("\n🎉 TUTTI I CHECK SUPERATI — STEP0 + STEP0B È SANO")
    print("👉 Puoi procedere con step1_profeta_train")


if __name__ == "__main__":
    main()