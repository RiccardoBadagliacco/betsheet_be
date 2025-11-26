#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BACKTEST — CHECKLIST STATISTICA MULTIGOL FAVORITA (MG 1–4 / 1–5)

Obiettivo:
- Usare SOLO informazioni storiche (stessa stagione, stesso lato casa/trasferta)
  per filtrare le partite dove la FAVORITA ha alta probabilità di:
    * MG 1–4 (1 <= gol favorita <= 4)
    * MG 1–5 (gol favorita >= 1)

Approccio:
- Loop cronologico sul dataset wide
- Per ogni match:
    * determina favorita (bk_p1/bk_p2) e prob implicita
    * se prob ∈ [1/1.90, 1/1.15] → match “in contesto MG”
    * usa SOLO lo storico precedente (team, season, side) per:
        - calcolare metriche offensive della favorita
        - calcolare metriche difensive dell’avversaria
        - applicare la checklist statistica
    * se la checklist passa → conteggia esito MG 1–4 e MG 1–5

Output:
- Baseline sul contesto MG (solo filtro quote)
- Risultati dopo checklist
- Copertura checklist + uplift

Esecuzione:
    python backtest_mg_checklist_stat.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Deque, Optional
from collections import defaultdict, deque

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PATH DATASET
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "correlazioni_affini_v2" / "data"
WIDE_FILE = DATA_DIR / "step4a_affini_index_wide_v2.parquet"


# ---------------------------------------------------------
# RANGE QUOTE FAVORITA
# ---------------------------------------------------------

# Range di quote desiderato per la favorita (1.15–1.90)
# bk_p1/bk_p2 = probabilità implicite, quindi:
#   quota 1.90 → p ≈ 0.5263
#   quota 1.15 → p ≈ 0.8696
FAV_PROB_MIN = 1.0 / 1.90
FAV_PROB_MAX = 1.0 / 1.15


# ---------------------------------------------------------
# STRUTTURE PER STORICO
# ---------------------------------------------------------

@dataclass
class FavHistory:
    """
    Storico offensivo per (team, season, side).

    Memorizza i gol fatti in tutte le partite precedenti (stessa stagione, stesso lato).
    """
    goals: list = field(default_factory=list)
    last5: Deque[int] = field(default_factory=lambda: deque(maxlen=5))

    def update(self, g: int) -> None:
        self.goals.append(int(g))
        self.last5.append(int(g))

    def metrics(self) -> Optional[dict]:
        if len(self.goals) == 0:
            return None

        arr = np.array(self.goals, dtype=float)
        n = len(arr)

        mean_g = float(arr.mean())
        var_g = float(arr.var()) if n > 1 else 0.0
        median_g = float(np.median(arr))

        zero_mask = (arr == 0)
        fourplus_mask = (arr >= 4)
        ge1_mask = (arr >= 1)
        in13_mask = (arr >= 1) & (arr <= 3)

        share_zero = zero_mask.mean()
        share_four = fourplus_mask.mean()
        share_ge1 = ge1_mask.mean()
        share_13 = in13_mask.mean()

        last5_list = list(self.last5)
        last5_ge1 = sum(g >= 1 for g in last5_list)

        return {
            "n": n,
            "mean_g": mean_g,
            "var_g": var_g,
            "median_g": median_g,
            "share_zero": share_zero,
            "share_four": share_four,
            "share_ge1": share_ge1,
            "share_13": share_13,
            "last5": last5_list,
            "last5_ge1": last5_ge1,
        }


@dataclass
class OppHistory:
    """
    Storico difensivo per (team, season, side).

    Memorizza i gol subiti in tutte le partite precedenti (stessa stagione, stesso lato).
    """
    conceded: list = field(default_factory=list)
    last5: Deque[int] = field(default_factory=lambda: deque(maxlen=5))

    def update(self, c: int) -> None:
        self.conceded.append(int(c))
        self.last5.append(int(c))

    def metrics(self) -> Optional[dict]:
        if len(self.conceded) == 0:
            return None

        arr = np.array(self.conceded, dtype=float)
        n = len(arr)

        mean_c = float(arr.mean())
        median_c = float(np.median(arr))

        fourplus_mask = (arr >= 4)
        share_four = fourplus_mask.mean()

        last5_list = list(self.last5)
        last5_ge1 = sum(g > 0 for g in last5_list)  # partite con almeno 1 gol subito

        return {
            "n": n,
            "mean_c": mean_c,
            "median_c": median_c,
            "share_four": share_four,
            "last5": last5_list,
            "last5_ge1": last5_ge1,
        }


# ---------------------------------------------------------
# FUNZIONI DI SUPPORTO
# ---------------------------------------------------------

def determine_favorite(row) -> Optional[Tuple[str, str, float, int, int]]:
    """
    Determina favorita a partire da bk_p1/bk_p2.

    Ritorna:
        (side, fav_team, fav_prob, fav_goals, opp_goals)
    oppure None se non determinabile.
    """
    try:
        p1 = float(row["bk_p1"])
        p2 = float(row["bk_p2"])
    except Exception:
        return None

    if pd.isna(p1) or pd.isna(p2):
        return None

    if p1 > p2:
        return ("home", row["home_team"], p1, int(row["home_ft"]), int(row["away_ft"]))
    elif p2 > p1:
        return ("away", row["away_team"], p2, int(row["away_ft"]), int(row["home_ft"]))
    else:
        return None


def get_season(row) -> str:
    """
    Usa la colonna 'season' se esiste, altrimenti l'anno della data.
    """
    if "season" in row.index:
        return str(row["season"])
    dt = pd.to_datetime(row["date"])
    return str(dt.year)


def has_two_consecutive_clean_sheets(last5: list) -> bool:
    """
    True se negli ultimi 5 match ci sono almeno due clean sheet consecutivi (0 gol subiti).
    """
    if len(last5) < 2:
        return False
    for i in range(1, len(last5)):
        if last5[i] == 0 and last5[i - 1] == 0:
            return True
    return False


# ---------------------------------------------------------
# CHECKLIST STATISTICA
# ---------------------------------------------------------

def passes_stat_checklist(fav_m: dict, opp_m: dict) -> bool:
    """
    Applica la checklist statistica:

    FAVORITA:
      - almeno 5 match storici
      - media gol fatti >= 1.4
      - % partite con almeno 1 gol >= 75%
      - % partite con 0 gol <= 20%
      - % partite con 1–3 gol >= 60%
      - % partite con 4+ gol <= 25%
      - ultime 5: almeno 3 partite con 1+ gol fatti
      - varianza gol <= 2.5
      - mediana gol fatti >= 1

    AVVERSARIA:
      - almeno 5 match storici
      - media gol subiti >= 1.0
      - % partite con 4+ gol subiti <= 20%
      - ultime 5: almeno 4 partite con gol subiti
      - nessuna coppia di clean sheet consecutivi negli ultimi 5

    NOTE:
      - Se non ci sono abbastanza partite (n < 5) → False.
    """
    # Favorita
    if fav_m is None or fav_m["n"] < 5:
        return False

    if fav_m["mean_g"] < 1.4:
        return False
    if fav_m["share_ge1"] < 0.75:
        return False
    if fav_m["share_zero"] > 0.20:
        return False
    if fav_m["share_13"] < 0.60:
        return False
    if fav_m["share_four"] > 0.25:
        return False
    if fav_m["last5_ge1"] < 3:
        return False
    if fav_m["var_g"] > 2.5:
        return False
    if fav_m["median_g"] < 1.0:
        return False

    # Avversaria
    if opp_m is None or opp_m["n"] < 5:
        return False

    if opp_m["mean_c"] < 1.0:
        return False
    if opp_m["share_four"] > 0.20:
        return False
    if opp_m["last5_ge1"] < 4:
        return False
    if has_two_consecutive_clean_sheets(opp_m["last5"]):
        return False

    # Se tutte le condizioni sono soddisfatte
    return True


# ---------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------

def run_backtest(n_matches: Optional[int] = None) -> None:
    print(f"📥 Carico wide da: {WIDE_FILE}")
    df = pd.read_parquet(WIDE_FILE)

    # Colonne minime
    required_cols = [
        "match_id", "date",
        "home_team", "away_team",
        "home_ft", "away_ft",
        "bk_p1", "bk_p2",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Mancano colonne nel wide: {missing}")

    # Filtra status ok se esiste
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if n_matches is not None and n_matches < len(df):
        df = df.iloc[:n_matches].copy()
        print(f"✂️  Uso solo i primi {len(df)} match.")
    else:
        print(f"➡️  Backtest su {len(df)} match totali.")

    # Storici per (team, season, side)
    fav_hist: Dict[Tuple[str, str, str], FavHistory] = defaultdict(FavHistory)
    opp_hist: Dict[Tuple[str, str, str], OppHistory] = defaultdict(OppHistory)

    # Contatori baseline (solo filtro quota)
    ctx_total = 0
    ctx_mg14_hits = 0  # 1–4
    ctx_mg15_hits = 0  # >=1

    # Contatori checklist
    sel_total = 0
    sel_mg14_hits = 0
    sel_mg15_hits = 0

    # Optional: breakdown home/away
    sel_home = sel_home_mg14 = sel_home_mg15 = 0
    sel_away = sel_away_mg14 = sel_away_mg15 = 0

    # Loop cronologico
    for _, row in df.iterrows():
        # skip match senza risultato
        if pd.isna(row["home_ft"]) or pd.isna(row["away_ft"]):
            continue

        fav_info = determine_favorite(row)
        if fav_info is not None:
            fav_side, fav_team, fav_prob, fav_goals, opp_goals = fav_info

            # Contesto MG: prob in range [1/1.90, 1/1.15]
            if FAV_PROB_MIN <= fav_prob <= FAV_PROB_MAX:
                ctx_total += 1

                # Baseline
                if 1 <= fav_goals <= 4:
                    ctx_mg14_hits += 1
                if fav_goals >= 1:
                    ctx_mg15_hits += 1

                # Chiavi storici (prima dell'update!)
                season = get_season(row)
                fav_key = (fav_team, season, fav_side)
                opp_team = row["away_team"] if fav_side == "home" else row["home_team"]
                opp_side = "away" if fav_side == "home" else "home"
                opp_key = (opp_team, season, opp_side)

                fh = fav_hist.get(fav_key)
                oh = opp_hist.get(opp_key)

                fav_m = fh.metrics() if fh is not None else None
                opp_m = oh.metrics() if oh is not None else None

                if passes_stat_checklist(fav_m, opp_m):
                    sel_total += 1

                    # MG 1–4
                    if 1 <= fav_goals <= 4:
                        sel_mg14_hits += 1
                    # MG 1–5
                    if fav_goals >= 1:
                        sel_mg15_hits += 1

                    # breakdown home/away
                    if fav_side == "home":
                        sel_home += 1
                        if 1 <= fav_goals <= 4:
                            sel_home_mg14 += 1
                        if fav_goals >= 1:
                            sel_home_mg15 += 1
                    else:
                        sel_away += 1
                        if 1 <= fav_goals <= 4:
                            sel_away_mg14 += 1
                        if fav_goals >= 1:
                            sel_away_mg15 += 1

        # --------------------------------------------
        # UPDATE STORICI DOPO aver usato il match
        # --------------------------------------------
        season = get_season(row)
        h_team = row["home_team"]
        a_team = row["away_team"]
        h_gf = int(row["home_ft"])
        a_gf = int(row["away_ft"])

        # offensivo
        fav_hist[(h_team, season, "home")].update(h_gf)
        fav_hist[(a_team, season, "away")].update(a_gf)

        # difensivo (gol subiti)
        opp_hist[(h_team, season, "home")].update(a_gf)
        opp_hist[(a_team, season, "away")].update(h_gf)

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    print("\n==================== RISULTATI CHECKLIST STATISTICA MG ====================\n")

    print(f"Totale match nel dataset: {len(df)}")
    print(f"Match nel CONTEXT MG (favorita quota 1.15–1.90): {ctx_total}")
    print(f"Match che PASSANO la checklist statistica:       {sel_total}")
    if ctx_total > 0:
        print(f"Copertura checklist: {sel_total}/{ctx_total} = {sel_total/ctx_total:.3f} "
              f"({(sel_total/ctx_total)*100:.1f}%)")
    print()

    if ctx_total == 0:
        print("❌ Nessun match nel contesto MG. Controlla bk_p1/bk_p2 e range quote.")
        print("\n🏁 BACKTEST TERMINATO.\n")
        return

    # Baseline
    base_mg14_rate = ctx_mg14_hits / ctx_total
    base_mg15_rate = ctx_mg15_hits / ctx_total

    print("🔹 BASELINE (solo filtro quota, nessuna checklist)")
    print(f"MG Favorita 1–4: {ctx_mg14_hits}/{ctx_total} = {base_mg14_rate:.3f} "
          f"({base_mg14_rate*100:.1f}%)")
    print(f"MG Favorita 1–5: {ctx_mg15_hits}/{ctx_total} = {base_mg15_rate:.3f} "
          f"({base_mg15_rate*100:.1f}%)")
    print()

    # Checklist
    print("🔹 CHECKLIST STATISTICA")
    if sel_total > 0:
        sel_mg14_rate = sel_mg14_hits / sel_total
        sel_mg15_rate = sel_mg15_hits / sel_total

        uplift_mg14 = sel_mg14_rate - base_mg14_rate
        uplift_mg15 = sel_mg15_rate - base_mg15_rate

        print(f"MG Favorita 1–4: {sel_mg14_hits}/{sel_total} = {sel_mg14_rate:.3f} "
              f"({sel_mg14_rate*100:.1f}%)")
        print(f"MG Favorita 1–5: {sel_mg15_hits}/{sel_total} = {sel_mg15_rate:.3f} "
              f"({sel_mg15_rate*100:.1f}%)")
        print()
        print(f"Uplift MG 1–4 vs baseline: {uplift_mg14:+.3f} ({uplift_mg14*100:+.1f} pp)")
        print(f"Uplift MG 1–5 vs baseline: {uplift_mg15:+.3f} ({uplift_mg15*100:+.1f} pp)")
    else:
        print("Nessun match soddisfa la checklist statistica.")

    # Breakdown home/away
    if sel_total > 0:
        print("\n🔹 BREAKDOWN LATO FAVORITA (solo match checklist=TRUE)")
        if sel_home > 0:
            rate_h_14 = sel_home_mg14 / sel_home
            rate_h_15 = sel_home_mg15 / sel_home
            print(f"- Favorita in CASA: {sel_home} match")
            print(f"    MG 1–4: {sel_home_mg14}/{sel_home} = {rate_h_14:.3f} "
                  f"({rate_h_14*100:.1f}%)")
            print(f"    MG 1–5: {sel_home_mg15}/{sel_home} = {rate_h_15:.3f} "
                  f"({rate_h_15*100:.1f}%)")
        else:
            print("- Favorita in CASA: 0 match")

        if sel_away > 0:
            rate_a_14 = sel_away_mg14 / sel_away
            rate_a_15 = sel_away_mg15 / sel_away
            print(f"- Favorita in TRASFERTA: {sel_away} match")
            print(f"    MG 1–4: {sel_away_mg14}/{sel_away} = {rate_a_14:.3f} "
                  f"({rate_a_14*100:.1f}%)")
            print(f"    MG 1–5: {sel_away_mg15}/{sel_away} = {rate_a_15:.3f} "
                  f"({rate_a_15*100:.1f}%)")
        else:
            print("- Favorita in TRASFERTA: 0 match")

    print("\n🏁 BACKTEST COMPLETATO.\n")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    # se vuoi limitare i match: run_backtest(n_matches=20000)
    run_backtest(n_matches=None)


if __name__ == "__main__":
    main()