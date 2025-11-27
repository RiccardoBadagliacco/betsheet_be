#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# Base directory of the correlazioni_affini_v2 data folder
BASE = Path(__file__).resolve().parents[1] / "data"

STEP0 = BASE / "step0_dataset_base.parquet"
STEP1A = BASE / "step1a_elo.parquet"
STEP1B = BASE / "step1b_form_recent.parquet"
STEP1C = BASE / "step1c_dataset_with_elo_form.parquet"
STEP2A = BASE / "step2a_features_with_picchetto_fix.parquet"


def header(title):
    print("\n" + "=" * 60)
    print("🔍 " + title)
    print("=" * 60)


def ok(msg):
    print(f"   ✅ {msg}")


def fail(msg):
    print(f"   ❌ {msg}")


def load():
    print("📥 Carico dataset...")
    df0 = pd.read_parquet(STEP0)
    df_elo = pd.read_parquet(STEP1A)
    df_form = pd.read_parquet(STEP1B)
    df1c = pd.read_parquet(STEP1C)
    df2 = pd.read_parquet(STEP2A)

    print("   ✔ Dati caricati.")
    return df0, df_elo, df_form, df1c, df2


# ============================================================
# TEST 1 — Fixture hanno form_fx > 0
# ============================================================
def test_form_fx(df1c):
    header("TEST 1 — Fixture hanno FORM FX valida")

    fx = df1c[df1c["home_ft"].isna()].copy()

    if fx.empty:
        fail("Nessuna fixture trovata in step1c")
        return False

    sample = fx[[
        "match_id", "home_team", "away_team",
        "home_form_pts_avg_lastN_fx",
        "away_form_pts_avg_lastN_fx"
    ]]

    if (sample["home_form_pts_avg_lastN_fx"] > 0).any():
        ok("Le fixture hanno form_fx > 0 (OK)")
        print(sample.head(5))
        return True
    else:
        fail("Tutte le fixture hanno form_fx = 0 → compute_fixture_form NON funziona")
        print(sample.head(5))
        return False


# ============================================================
# TEST 2 — Elo fixture corretto
# ============================================================
def test_elo_fx(df1c, df_elo):
    header("TEST 2 — Elo fixture coerente con la storia")

    fx = df1c[df1c["home_ft"].isna()].copy()
    fx = fx.sort_values("date")

    if fx.empty:
        fail("Nessuna fixture trovata.")
        return False

    f = fx.iloc[0]
    team = f["home_team"]
    date_fx = f["date"]

    hist = df_elo[
        ((df_elo["home_team"] == team) | (df_elo["away_team"] == team))
        & (df_elo["date"] < date_fx)
    ].sort_values("date")

    if hist.empty:
        fail(f"Nessun match storico trovato per {team}")
        return False

    last = hist.iloc[-1]
    expected = last["elo_home_post"] if last["home_team"] == team else last["elo_away_post"]

    actual = f["elo_home_pre"]   # ✔ FIX QUI

    print(f"Team: {team}")
    print(f"Elo fixture:  {actual}")
    print(f"Elo expected: {expected}")

    if np.isclose(actual, expected, atol=1e-6):
        ok("Elo fixture corretto (OK)")
        return True
    else:
        fail("Elo fixture NON corrisponde alla storia → compute_fixture_elo rotto")
        return False

# ============================================================
# TEST 3 — Form fixture deriva SOLO da date < fixture
# ============================================================
def test_form_history(df1c, df_elo):
    header("TEST 3 — Form FX deriva da storico precedente")

    fx = df1c[df1c["home_ft"].isna()].copy()
    if fx.empty:
        fail("Nessuna fixture trovata.")
        return False

    f = fx.iloc[0]
    team = f["home_team"]
    date_fx = f["date"]

    # Usa df1c per la storia, NON df_form
    hist = df1c[
        (df1c["home_team"] == team) &
        (df1c["date"] < date_fx) &
        (df1c["home_ft"].notna())  # solo match storici
    ].sort_values("date")

    if hist.empty:
        fail(f"Nessun match storico trovato per {team}")
        return False

    print(
        hist.tail(5)[[
            "match_id", "date",
            "home_team",
            "home_form_pts_avg_lastN",
            "home_form_gf_avg_lastN"
        ]]
    )

    if (hist["date"] < date_fx).all():
        ok("Tutte le date dello storico sono < fixture (OK)")
        return True
    else:
        fail("Alcune date storiche sono >= fixture → LEAKAGE")
        return False

# ============================================================
# TEST 4 — Picchetto tecnico usa dati tecnici (non solo bookmaker)
# ============================================================
def test_picchetto(df2):
    header("TEST 4 — Picchetto tecnico funziona (no copier del bookmaker)")

    fx = df2[df2["home_ft"].isna()].copy()
    fx = fx.sort_values("date")

    if fx.empty:
        fail("Nessuna fixture trovata.")
        return False

    f = fx.iloc[0]

    print("bk_p1=", f["bk_p1"], "pic_p1=", f["pic_p1"])
    print("bk_pO25=", f["bk_pO25"], "pic_pO25=", f["pic_pO25"])

    if not np.isclose(f["pic_p1"], f["bk_p1"], atol=1e-3):
        ok("pic_p1 diverge da bk_p1 → OK, tecnico attivo")
    else:
        fail("pic_p1 = bk_p1 → tecnico NON si attiva")
        return False

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    df0, df_elo, df_form, df1c, df2 = load()

    results = {
        "FORM_FX": test_form_fx(df1c),
        "ELO_FX": test_elo_fx(df1c, df_elo),
        "FORM_HISTORY": test_form_history(df1c, df_elo),
        "PICCHETTO": test_picchetto(df2),
    }

    print("\n🧪 RISULTATI FINALI")
    for k, v in results.items():
        print(f"   {k}: {'OK ✔' if v else 'FAIL ❌'}")
