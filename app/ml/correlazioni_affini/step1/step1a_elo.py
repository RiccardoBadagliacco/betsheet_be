import json
import pandas as pd
from pathlib import Path

# ===================================================
# PARAMETERS
# ===================================================

K = 20
HOME_ADV = 75     # come nel vecchio script
ELO_START = 1500


# ===================================================
# PATHS
# ===================================================

BASE_DIR = Path(__file__).resolve().parents[2]       # → app/ml
PROJECT_ROOT = BASE_DIR.parents[1]                   # → root progetto
AFFINI_DIR = BASE_DIR / "correlazioni_affini"
DATA_DIR = AFFINI_DIR / "data"

RAW_JSON = PROJECT_ROOT / "data" / "all_matches_raw.json"
OUTPUT_FILE = DATA_DIR / "step1a_elo.parquet"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================
# LOAD RAW MATCHES (IDENTICO ALLO SCRIPT ORIGINALE)
# ===================================================

def load_raw_matches(path: Path) -> pd.DataFrame:
    print("📥 Carico dati RAW...")
    with open(path, "r") as f:
        data = json.load(f)

    rows = []

    for match_id, wrapper in data.items():
        if "match" not in wrapper:
            continue

        m = wrapper["match"]

        date = m.get("date")
        home = m.get("home_team", {}).get("name")
        away = m.get("away_team", {}).get("name")

        res = m.get("result") or {}
        home_ft = res.get("home_ft")
        away_ft = res.get("away_ft")

        if home_ft is None or away_ft is None:
            continue

        rows.append({
            "match_id": match_id,
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_ft": home_ft,
            "away_ft": away_ft,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ===================================================
# ELO FUNZIONI (IDENTICHE ALLO SCRIPT ORIGINALE)
# ===================================================

def expected_score(elo_A, elo_B):
    return 1 / (1 + 10 ** ((elo_B - elo_A) / 400))


def update_elo(elo_home, elo_away, home_goals, away_goals):
    if home_goals > away_goals:
        S_home, S_away = 1, 0
    elif home_goals < away_goals:
        S_home, S_away = 0, 1
    else:
        S_home, S_away = 0.5, 0.5

    E_home = expected_score(elo_home + HOME_ADV, elo_away)
    E_away = expected_score(elo_away, elo_home + HOME_ADV)

    new_home = elo_home + K * (S_home - E_home)
    new_away = elo_away + K * (S_away - E_away)

    return new_home, new_away


# ===================================================
# MAIN ELO COMPUTATION (IDENTICO ALL’ORIGINALE)
# ===================================================

def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    print("⚡ Calcolo Elo come nella versione originale...")

    teams = {}
    records = []

    for _, r in df.iterrows():
        ht = r.home_team
        at = r.away_team

        elo_h = teams.get(ht, ELO_START)
        elo_a = teams.get(at, ELO_START)

        new_h, new_a = update_elo(elo_h, elo_a, r.home_ft, r.away_ft)

        teams[ht] = new_h
        teams[at] = new_a

        records.append({
            "match_id": r.match_id,
            "date": r.date,
            "home_team": ht,
            "away_team": at,
            "home_ft": r.home_ft,
            "away_ft": r.away_ft,
            "elo_home_pre": elo_h,
            "elo_away_pre": elo_a,
            "elo_home_post": new_h,
            "elo_away_post": new_a,
            "elo_diff": elo_h - elo_a,
        })

    return pd.DataFrame(records)


# ===================================================
# MAIN WRAPPER
# ===================================================

def main():
    print(f"📥 Carico RAW → {RAW_JSON}")
    df = load_raw_matches(RAW_JSON)

    print("⚙️ Computo ELO...")
    df_elo = compute_elo(df)

    print(f"💾 Salvo Elo → {OUTPUT_FILE}")
    df_elo.to_parquet(OUTPUT_FILE, index=False)

    print("✅ STEP1A COMPLETATO!")


if __name__ == "__main__":
    main()