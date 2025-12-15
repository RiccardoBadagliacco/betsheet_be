# ============================================================
# app/ml/profeta/step2_profeta_single.py
# ============================================================

"""
Predizione singola usando PROFETA V0.

Uso:
    python step2_profeta_single.py --match_id <uuid>
    python step2_profeta_single.py --fixture_id <uuid>
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR / "data"
STEP0_PATH = DATA_DIR / "step0_profeta.parquet"

# Importa engine Profeta
from app.ml.profeta.step2_profeta import ProfetaEngine


# ------------------------------------------------------------
# Helpers semaforo confidenza
# ------------------------------------------------------------

def prob_to_semaforo(p: float) -> str:
    """
    Ritorna un'iconcina + etichetta in base alla probabilità.
    Soglie basate sui pattern trovati nello STEP5.
    """
    if p is None:
        return "⚪ n/d"
    try:
        p = float(p)
    except Exception:
        return "⚪ n/d"

    if p < 0.60:
        return "🔴 bassa"
    elif p < 0.70:
        return "🟠 medio-bassa"
    elif p < 0.80:
        return "🟢 alta"
    elif p < 0.90:
        return "🟢💪 molto alta"
    else:
        return "🔵 quasi certa"


def fmt_prob(p: float) -> str:
    """Formatta la probabilità con percentuale + semaforo."""
    if p is None:
        return "n/d (⚪)"
    return f"{p:.1%}  {prob_to_semaforo(p)}"


def pretty_print_markets(markets):
    print("\n🎯 1X2")
    print(f"   1 = {fmt_prob(markets['p1'])}")
    print(f"   X = {fmt_prob(markets['px'])}")
    print(f"   2 = {fmt_prob(markets['p2'])}")

    print("\n🤝 GG / NoGG")
    print(f"   GG   = {fmt_prob(markets['p_gg'])}")
    print(f"   NoGG = {fmt_prob(markets['p_nogg'])}")

    print("\n📦 Over/Under")
    print(f"   Over 1.5  = {fmt_prob(markets['p_over_1_5'])}")
    print(f"   Over 2.5  = {fmt_prob(markets['p_over_2_5'])}")
    print(f"   Over 3.5  = {fmt_prob(markets['p_over_3_5'])}")
    print(f"   Under 1.5 = {fmt_prob(markets['p_under_1_5'])}")
    print(f"   Under 2.5 = {fmt_prob(markets['p_under_2_5'])}")
    print(f"   Under 3.5 = {fmt_prob(markets['p_under_3_5'])}")

    print("\n📦 Multigol Totali")
    for k, label in [
        ("p_mg_1_3", "MG 1–3"),
        ("p_mg_1_4", "MG 1–4"),
        ("p_mg_1_5", "MG 1–5"),
        ("p_mg_2_4", "MG 2–4"),
        ("p_mg_2_5", "MG 2–5"),
        ("p_mg_2_6", "MG 2–6"),
    ]:
        print(f"   {label:<7} = {fmt_prob(markets[k])}")

    print("\n📦 Multigol Casa")
    for k, label in [
        ("p_mg_home_1_3", "Casa 1–3"),
        ("p_mg_home_1_4", "Casa 1–4"),
        ("p_mg_home_1_5", "Casa 1–5"),
    ]:
        print(f"   {label:<8} = {fmt_prob(markets[k])}")

    print("\n📦 Multigol Ospite")
    for k, label in [
        ("p_mg_away_1_3", "Ospite 1–3"),
        ("p_mg_away_1_4", "Ospite 1–4"),
        ("p_mg_away_1_5", "Ospite 1–5"),
    ]:
        print(f"   {label:<9} = {fmt_prob(markets[k])}")

    print("\n🎯 Top 6 Risultati esatti:")
    for score, p in markets["top6_correct_score"]:
        # qui le probabilità sono piccole, il semaforo ha meno senso
        print(f"   {score}: {p:.2%}")


def load_row_from_step0(match_id=None, fixture_id=None):
    df = pd.read_parquet(STEP0_PATH)

    if match_id:
        row = df[(df["match_id"] == match_id) & (df["is_fixture"] == False)]
        if len(row):
            return row.iloc[0]

    if fixture_id:
        row = df[(df["match_id"] == fixture_id) & (df["is_fixture"] == True)]
        if len(row):
            return row.iloc[0]

    print("❌ Nessuna partita trovata con ID indicato.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_id", type=str, help="ID match storico")
    parser.add_argument("--fixture_id", type=str, help="ID fixture futura")
    args = parser.parse_args()

    if not args.match_id and not args.fixture_id:
        print("⚠️ Devi specificare --match_id oppure --fixture_id")
        sys.exit(1)

    row = load_row_from_step0(args.match_id, args.fixture_id)

    engine = ProfetaEngine(max_goals=10)
    result = engine.predict_from_row(row)

    lam_h = result["lambda_home"]
    lam_a = result["lambda_away"]
    markets = result["markets"]

    print("\n===================================================")
    print("🔮 PROFETA V0 — ANALISI SINGOLA PARTITA")
    print("===================================================")

    print(f"\n📌 Match ID: {row['match_id']}")
    print(f"📌 Fixture? {row['is_fixture']}")
    print(f"🏟  Lega: {row['league_id']}  —  Stagione: {row['season_id']}")
    print(f"👕 Home team:  {row['home_team_id']}")
    print(f"👕 Away team:  {row['away_team_id']}")

    print("\n⚙️ Lambda values (expected goals)")
    print(f"   λ_home = {lam_h:.3f}")
    print(f"   λ_away = {lam_a:.3f}")
    print(f"   xG Tot = {lam_h + lam_a:.3f}")

    pretty_print_markets(markets)

    print("\n===================================================\n")


if __name__ == "__main__":
    main()