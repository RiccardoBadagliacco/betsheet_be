# ============================================================
# app/ml/correlazioni_affini_v2/step0/step0_dataset_base_features.py
# ============================================================

"""
STEP0 v2 — Dataset base tecnico (solo mercato + risultato)

Obiettivo:
    Costruire un dataset base PULITO e REALISTICO che contenga:
      - Metadati match (lega, stagione, squadre)
      - Quote medie 1X2 + OU 2.5
      - Probabilità implicite e overround
      - Probabilità normalizzate (bk_p*)
      - Fair odds (1 / bk_p*)
      - Misure di "shape" del mercato (entropia 1X2 / OU2.5)
      - Risultato reale + flag 1X2 e OU (0.5 / 1.5 / 2.5 / 3.5 solo da gol reali)

N.B.:
    - NON inseriamo colonne OU1.5 e BTTS di mercato, perché nel DB sono vuote.
    - Tutte le feature "intelligenti" (elo, forma, lambda, att/def strength, ecc.)
      verranno aggiunte in STEP1.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

# ------------------------------------------------------------
# FIX PATH: aggiunge la root del progetto per importare "app."
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from app.db.database_football import get_football_db
from app.db.models_football import Match, Season, League


# ------------------------------------------------------------
# PATHS OUTPUT
# ------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

OUT_PATH = DATA_DIR / "step0_dataset_base.parquet"


# ============================================================
# UTILITIES
# ============================================================

def safe_div_odds_to_prob(odds: Optional[float]) -> Optional[float]:
    """
    Converte odds in probabilità implicita (1/odds).
    Ritorna None se odds non valida.
    """
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if o <= 1e-9:
        return None
    return 1.0 / o


def normalize_three(a: Optional[float],
                    b: Optional[float],
                    c: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Normalizza tre probabilità grezze in modo che sommino a 1.
    Ritorna (p1_norm, p2_norm, p3_norm, somma_raw).
    """
    if a is None or b is None or c is None:
        return None, None, None, None
    s = a + b + c
    if s <= 0:
        return None, None, None, s
    return a / s, b / s, c / s, s


def normalize_two(a: Optional[float],
                  b: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Normalizza due probabilità grezze (es. OU2.5) in modo che sommino a 1.
    Ritorna (pA_norm, pB_norm, somma_raw).
    """
    if a is None or b is None:
        return None, None, None
    s = a + b
    if s <= 0:
        return None, None, s
    return a / s, b / s, s


def normalize_season(s: Optional[str]) -> Optional[str]:
    """
    Normalizza il formato della season, es. "2023_2024" -> "2023/2024".
    """
    if s is None:
        return None
    return str(s).replace("_", "/")


def entropy_probs(probs: List[float]) -> Optional[float]:
    """
    Entropia normalizzata di una lista di probabilità.
    - probs devono sommare ~1 (ma non lo imponiamo rigidamente)
    - ritorna valore in [0,1] (0 = esito certo, 1 = massima incertezza)
    """
    arr = np.array([p for p in probs if p is not None and p > 0], dtype=float)
    if len(arr) == 0:
        return None
    arr = arr / arr.sum()
    h = -np.sum(arr * np.log(arr))
    h_max = np.log(len(arr))
    if h_max <= 0:
        return None
    return float(h / h_max)


def detect_fav_1x2(p1: Optional[float], px: Optional[float], p2: Optional[float]) -> Tuple[Optional[str], Optional[float]]:
    """
    Identifica il lato favorito sull'1X2 e la probabilità associata.
    Ritorna (side, prob) con side in {"home","draw","away"}.
    """
    if p1 is None or px is None or p2 is None:
        return None, None
    arr = np.array([p1, px, p2], dtype=float)
    idx = int(arr.argmax())
    side = ["home", "draw", "away"][idx]
    return side, float(arr[idx])


# ============================================================
# MAIN LOGIC
# ============================================================

def build_step0_dataset() -> pd.DataFrame:
    """
    Legge i match dal DB e costruisce dataset base tecnico v2.
    """
    session: Session = next(get_football_db())
    rows = []

    print("📥 Lettura MATCH dal database…")

    q = (
        session.query(Match)
        .join(Season, Match.season_id == Season.id)
        .join(League, Season.league_id == League.id)
    )

    matches: List[Match] = q.all()
    print(f"   → Trovati {len(matches)} match nel DB")

    for m in matches:
        league: League = m.season.league if m.season else None
        country = getattr(league, "country", None)

        # ----------------------------------------------------
        # 1X2 — PROBABILITÀ IMPLICITE E OVERROUND
        # ----------------------------------------------------
        avg_home_odds = m.avg_home_odds
        avg_draw_odds = m.avg_draw_odds
        avg_away_odds = m.avg_away_odds

        p1_raw = safe_div_odds_to_prob(avg_home_odds)
        px_raw = safe_div_odds_to_prob(avg_draw_odds)
        p2_raw = safe_div_odds_to_prob(avg_away_odds)

        # Richiediamo almeno quote 1X2 valide
        if p1_raw is None or px_raw is None or p2_raw is None:
            continue

        bk_p1, bk_px, bk_p2, bk_sum_1x2_raw = normalize_three(p1_raw, px_raw, p2_raw)
        if bk_p1 is None:
            continue

        bk_overround_1x2 = bk_sum_1x2_raw - 1.0 if bk_sum_1x2_raw is not None else None

        # Fair odds da probabilità normalizzate
        fair_home_odds = 1.0 / bk_p1 if bk_p1 and bk_p1 > 0 else None
        fair_draw_odds = 1.0 / bk_px if bk_px and bk_px > 0 else None
        fair_away_odds = 1.0 / bk_p2 if bk_p2 and bk_p2 > 0 else None

        entropy_1x2 = entropy_probs([bk_p1, bk_px, bk_p2])
        fav_side_1x2, fav_prob_1x2 = detect_fav_1x2(bk_p1, bk_px, bk_p2)

        is_big_fav_home = 1 if (fav_side_1x2 == "home" and fav_prob_1x2 is not None and fav_prob_1x2 >= 0.60) else 0
        is_big_fav_away = 1 if (fav_side_1x2 == "away" and fav_prob_1x2 is not None and fav_prob_1x2 >= 0.60) else 0

        # ----------------------------------------------------
        # OU 2.5 — PROBABILITÀ E OVERROUND (se disponibili)
        # ----------------------------------------------------
        avg_over_25_odds = m.avg_over_25_odds
        avg_under_25_odds = m.avg_under_25_odds

        pO25_raw = safe_div_odds_to_prob(avg_over_25_odds)
        pU25_raw = safe_div_odds_to_prob(avg_under_25_odds)

        bk_pO25 = bk_pU25 = bk_sum_ou25_raw = bk_overround_ou25 = None
        entropy_ou25 = None

        if pO25_raw is not None and pU25_raw is not None:
            bk_pO25, bk_pU25, bk_sum_ou25_raw = normalize_two(pO25_raw, pU25_raw)
            if bk_sum_ou25_raw is not None:
                bk_overround_ou25 = bk_sum_ou25_raw - 1.0
            entropy_ou25 = entropy_probs([bk_pO25, bk_pU25])

        # ----------------------------------------------------
        # RISULTATO REALE
        # ----------------------------------------------------
        home_ft = m.home_goals_ft
        away_ft = m.away_goals_ft

        # se mancano i gol, saltiamo (perché clustering affini userà solo match chiusi)
        if home_ft is None or away_ft is None:
            continue

        total_goals = float(home_ft) + float(away_ft)

        is_home_win = int(home_ft > away_ft)
        is_draw = int(home_ft == away_ft)
        is_away_win = int(home_ft < away_ft)

        is_over05 = int(total_goals >= 1)
        is_over15 = int(total_goals >= 2)
        is_over25 = int(total_goals >= 3)
        is_over35 = int(total_goals >= 4)

        is_under25 = int(total_goals < 3)

        rows.append(
            {
                # ------------------
                # Metadati
                # ------------------
                "match_id": str(m.id),
                "date": str(m.match_date) if m.match_date is not None else None,
                "league": getattr(league, "code", None),
                "country": getattr(country, "code", None),
                "season": normalize_season(getattr(m.season, "name", None)),
                "home_team": m.home_team.name if m.home_team else None,
                "away_team": m.away_team.name if m.away_team else None,

                # ------------------
                # Quote 1X2 raw
                # ------------------
                "avg_home_odds": float(avg_home_odds) if avg_home_odds is not None else None,
                "avg_draw_odds": float(avg_draw_odds) if avg_draw_odds is not None else None,
                "avg_away_odds": float(avg_away_odds) if avg_away_odds is not None else None,

                "p1_raw": p1_raw,
                "px_raw": px_raw,
                "p2_raw": p2_raw,
                "bk_sum_1x2_raw": bk_sum_1x2_raw,
                "bk_overround_1x2": bk_overround_1x2,

                "bk_p1": bk_p1,
                "bk_px": bk_px,
                "bk_p2": bk_p2,

                "fair_home_odds": fair_home_odds,
                "fair_draw_odds": fair_draw_odds,
                "fair_away_odds": fair_away_odds,

                "entropy_bk_1x2": entropy_1x2,
                "fav_side_1x2": fav_side_1x2,
                "fav_prob_1x2": fav_prob_1x2,
                "is_big_fav_home": is_big_fav_home,
                "is_big_fav_away": is_big_fav_away,

                # ------------------
                # OU 2.5 (se presenti)
                # ------------------
                "avg_over_25_odds": float(avg_over_25_odds) if avg_over_25_odds is not None else None,
                "avg_under_25_odds": float(avg_under_25_odds) if avg_under_25_odds is not None else None,
                "pO25_raw": pO25_raw,
                "pU25_raw": pU25_raw,
                "bk_sum_ou25_raw": bk_sum_ou25_raw,
                "bk_overround_ou25": bk_overround_ou25,
                "bk_pO25": bk_pO25,
                "bk_pU25": bk_pU25,
                "entropy_bk_ou25": entropy_ou25,

                # ------------------
                # Risultato reale
                # ------------------
                "home_ft": float(home_ft),
                "away_ft": float(away_ft),
                "total_goals": total_goals,

                "is_home_win": is_home_win,
                "is_draw": is_draw,
                "is_away_win": is_away_win,

                "is_over05": is_over05,
                "is_over15": is_over15,
                "is_over25": is_over25,
                "is_over35": is_over35,
                "is_under25": is_under25,
            }
        )

    df = pd.DataFrame(rows)
    print("📊 Dataset base tecnico creato:", df.shape)

    return df


def main():
    print("🚀 STEP0 v2 — Generazione dataset base tecnico (definitivo)")
    print(f"📦 Output previsto: {OUT_PATH}")

    df = build_step0_dataset()

    if df.empty:
        print("⚠️ Dataset vuoto, niente output.")
        return

    df.to_parquet(OUT_PATH, index=False)
    print("💾 Salvato:", OUT_PATH)


if __name__ == "__main__":
    main()