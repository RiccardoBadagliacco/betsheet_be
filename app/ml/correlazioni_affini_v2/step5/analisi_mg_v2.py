#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BACKTEST GRID POISSON — MULTIGOL FAVORITA (MG 1–4 / 1–5)

Questo script:
- legge il wide (step4a_affini_index_wide_v2.parquet)
- scorre cronologicamente le partite
- costruisce storici offensivi/difensivi per (team, season, side)
- per ogni match con FAVORITA (quota implicita 1.15–1.90) calcola:

    λ_att     = media gol fatti favorita (stesso lato, stessa stagione)
    λ_def_opp = media gol subiti avversaria (stesso lato, stessa stagione)
    λ_fav     = 0.6 * λ_att + 0.4 * λ_def_opp    (misto attacco+difesa opp)

    P_MG14 = P(1 <= G_fav <= 4)   con G_fav ~ Poisson(λ_fav)
    P_MG15 = P(G_fav >= 1)

- applica in parallelo 4 varianti di checklist Poisson:

    v2A: prudente, alta confidenza, copertura bassa
    v2B: più equilibrata (target ~10–12% copertura)
    v2C: larga (target ~15–20% copertura)
    v2D: ottimizzata per MG 1–4 (controlla anche P(G>=5))

- produce un report finale con:

    baseline MG 1–4 / 1–5 sul CONTEXT MG
    per ogni versione:
        - copertura
        - MG 1–4 e 1–5
        - uplift rispetto alla baseline
"""

import math
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
# RANGE QUOTE FAVORITA (stesso di prima)
# ---------------------------------------------------------

FAV_PROB_MIN = 1.0 / 1.90
FAV_PROB_MAX = 1.0 / 1.15


# ---------------------------------------------------------
# STRUTTURE STORICO
# ---------------------------------------------------------

@dataclass
class TeamSideHistory:
    """
    Storico per (team, season, side).

    Qui teniamo separato:
      - goals_for: gol fatti (per attacco)
      - goals_against: gol subiti (per difesa)
    """
    goals_for: list = field(default_factory=list)
    goals_against: list = field(default_factory=list)
    last5_for: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
    last5_against: Deque[int] = field(default_factory=lambda: deque(maxlen=5))

    def update(self, gf: int, ga: int) -> None:
        gf = int(gf)
        ga = int(ga)
        self.goals_for.append(gf)
        self.goals_against.append(ga)
        self.last5_for.append(gf)
        self.last5_against.append(ga)

    def attack_lambda(self) -> Optional[float]:
        if not self.goals_for:
            return None
        return float(np.mean(self.goals_for))

    def defense_lambda(self) -> Optional[float]:
        if not self.goals_against:
            return None
        return float(np.mean(self.goals_against))

    def n_matches(self) -> int:
        return len(self.goals_for)


# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------

def determine_favorite(row) -> Optional[Tuple[str, str, float, int, int]]:
    """
    Determina favorita a partire da bk_p1/bk_p2.

    Ritorna:
        (fav_side, fav_team, fav_prob, fav_goals, opp_goals)
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


def poisson_probs(lambda_fav: float) -> Dict[str, float]:
    """
    Calcola:
      - P0      = P(G=0)
      - P1_4    = P(1 <= G <= 4)
      - P_ge1   = P(G >= 1)
      - P_ge5   = P(G >= 5)
    con G ~ Poisson(λ).
    """
    if lambda_fav <= 0 or math.isnan(lambda_fav):
        return {"P0": math.nan, "P1_4": math.nan, "P_ge1": math.nan, "P_ge5": math.nan}

    # Poisson pmf k=0..4
    pmf = []
    for k in range(0, 5):
        pk = math.exp(-lambda_fav) * (lambda_fav ** k) / math.factorial(k)
        pmf.append(pk)

    P0 = pmf[0]
    P1_4 = sum(pmf[1:5])
    P_le4 = sum(pmf[0:5])
    P_ge1 = 1.0 - P0
    P_ge5 = 1.0 - P_le4
    return {
        "P0": P0,
        "P1_4": P1_4,
        "P_ge1": P_ge1,
        "P_ge5": P_ge5,
    }


# ---------------------------------------------------------
# CONFIGURAZIONI CHECKLIST v2 (Poisson)
# ---------------------------------------------------------

CHECKLIST_CONFIGS = {
    # Versione prudente, copertura bassa, confidenza alta
    "v2A": {
        "min_n_fav": 5,
        "min_n_opp": 5,
        "min_lambda_att": 1.4,
        "min_lambda_def_opp": 1.0,
        "lambda_fav_min": 1.6,
        "lambda_fav_max": 3.4,
        "min_P_MG15": 0.85,
        "min_P_MG14": 0.75,
        "max_P_ge5": None,   # nessun vincolo su goleade
    },
    # Versione equilibrata (target ~10–12% copertura)
    "v2B": {
        "min_n_fav": 4,
        "min_n_opp": 4,
        "min_lambda_att": 1.3,
        "min_lambda_def_opp": 0.9,
        "lambda_fav_min": 1.4,
        "lambda_fav_max": 3.6,
        "min_P_MG15": 0.80,
        "min_P_MG14": 0.72,
        "max_P_ge5": None,
    },
    # Versione larga (target ~15–20% copertura)
    "v2C": {
        "min_n_fav": 3,
        "min_n_opp": 3,
        "min_lambda_att": 1.2,
        "min_lambda_def_opp": 0.8,
        "lambda_fav_min": 1.3,
        "lambda_fav_max": 3.8,
        "min_P_MG15": 0.75,
        "min_P_MG14": 0.68,
        "max_P_ge5": None,
    },
    # Versione ottimizzata per MG 1–4 (controlla anche goleade 5+)
    "v2D": {
        "min_n_fav": 4,
        "min_n_opp": 4,
        "min_lambda_att": 1.3,
        "min_lambda_def_opp": 0.9,
        "lambda_fav_min": 1.3,
        "lambda_fav_max": 3.0,
        "min_P_MG15": 0.80,
        "min_P_MG14": 0.78,
        "max_P_ge5": 0.12,   # vogliamo poche partite con 5+ gol favorita
    },
}


def passes_config(
    cfg: Dict[str, float],
    n_fav: int,
    n_opp: int,
    lambda_att: Optional[float],
    lambda_def_opp: Optional[float],
    lambda_fav: Optional[float],
    P_MG14: float,
    P_MG15: float,
    P_ge5: float,
) -> bool:
    """
    Verifica se il match passa una certa configurazione checklist Poisson.
    """
    if lambda_att is None or lambda_def_opp is None or lambda_fav is None:
        return False

    # storico minimo
    if n_fav < cfg["min_n_fav"]:
        return False
    if n_opp < cfg["min_n_opp"]:
        return False

    # forme
    if lambda_att < cfg["min_lambda_att"]:
        return False
    if lambda_def_opp < cfg["min_lambda_def_opp"]:
        return False

    # range ragionevole di λ_fav
    if not (cfg["lambda_fav_min"] <= lambda_fav <= cfg["lambda_fav_max"]):
        return False

    # soglie Poisson
    if P_MG15 < cfg["min_P_MG15"]:
        return False
    if P_MG14 < cfg["min_P_MG14"]:
        return False

    # opzionale: limitare P(G>=5) per controllare gol fuori range 1–4
    max_P_ge5 = cfg.get("max_P_ge5", None)
    if max_P_ge5 is not None and P_ge5 > max_P_ge5:
        return False

    return True


# ---------------------------------------------------------
# BACKTEST GRID
# ---------------------------------------------------------

def run_backtest(n_matches: Optional[int] = None) -> None:
    print(f"📥 Carico wide da: {WIDE_FILE}")
    df = pd.read_parquet(WIDE_FILE)

    required_cols = [
        "match_id", "date",
        "home_team", "away_team",
        "home_ft", "away_ft",
        "bk_p1", "bk_p2",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Mancano colonne nel wide: {missing}")

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if n_matches is not None and n_matches < len(df):
        df = df.iloc[:n_matches].copy()
        print(f"✂️  Uso solo i primi {len(df)} match.")
    else:
        print(f"➡️  Backtest su {len(df)} match totali.")

    # Storico: un unico dict (team, season, side) -> TeamSideHistory
    history: Dict[Tuple[str, str, str], TeamSideHistory] = defaultdict(TeamSideHistory)

    # Contesto MG (solo filtro quota)
    ctx_total = 0
    ctx_mg14_hits = 0
    ctx_mg15_hits = 0

    # Risultati per ogni versione
    sel_total = {k: 0 for k in CHECKLIST_CONFIGS.keys()}
    sel_mg14_hits = {k: 0 for k in CHECKLIST_CONFIGS.keys()}
    sel_mg15_hits = {k: 0 for k in CHECKLIST_CONFIGS.keys()}

    # Per debug: salviamo qualche match a campione per le prime versioni
    debug_printed = False

    # Loop cronologico
    for _, row in df.iterrows():
        if pd.isna(row["home_ft"]) or pd.isna(row["away_ft"]):
            # se manca il risultato, aggiorniamo comunque storici? No: non ha senso
            continue

        fav_info = determine_favorite(row)
        if fav_info is None:
            # poi aggiorniamo storici a fine loop
            pass
        else:
            fav_side, fav_team, fav_prob, fav_goals, opp_goals = fav_info

            if FAV_PROB_MIN <= fav_prob <= FAV_PROB_MAX:
                ctx_total += 1

                # Baseline MG 1–4 / 1–5
                if 1 <= fav_goals <= 4:
                    ctx_mg14_hits += 1
                if fav_goals >= 1:
                    ctx_mg15_hits += 1

                season = get_season(row)

                # favorite (attacco)
                fav_key = (fav_team, season, fav_side)
                # avversaria lato opposto
                if fav_side == "home":
                    opp_team = row["away_team"]
                    opp_side = "away"
                else:
                    opp_team = row["home_team"]
                    opp_side = "home"
                opp_key = (opp_team, season, opp_side)

                fav_hist = history.get(fav_key, None)
                opp_hist = history.get(opp_key, None)

                lambda_att = fav_hist.attack_lambda() if fav_hist else None
                lambda_def_opp = opp_hist.defense_lambda() if opp_hist else None

                n_fav = fav_hist.n_matches() if fav_hist else 0
                n_opp = opp_hist.n_matches() if opp_hist else 0

                # λ favorita combinato (0.6*attacco + 0.4*difesa opp)
                if lambda_att is not None and lambda_def_opp is not None:
                    lambda_fav = 0.6 * lambda_att + 0.4 * lambda_def_opp
                    probs = poisson_probs(lambda_fav)
                    P_MG14 = probs["P1_4"]
                    P_MG15 = probs["P_ge1"]
                    P_ge5 = probs["P_ge5"]
                else:
                    lambda_fav = None
                    P_MG14 = math.nan
                    P_MG15 = math.nan
                    P_ge5 = math.nan

                # Per il primo match che passa almeno una versione, stampiamo debug
                if (not debug_printed) and lambda_fav is not None:
                    any_pass = False
                    for name, cfg in CHECKLIST_CONFIGS.items():
                        if passes_config(
                            cfg, n_fav, n_opp,
                            lambda_att, lambda_def_opp, lambda_fav,
                            P_MG14, P_MG15, P_ge5
                        ):
                            any_pass = True
                            break
                    if any_pass:
                        debug_printed = True
                        print("\n[DEBUG GRID - esempio match selezionato da almeno una versione]")
                        print(f"  Data: {row['date']}, {row['home_team']} vs {row['away_team']} ({fav_side} favorita)")
                        print(f"  λ_att={lambda_att:.3f}, λ_def_opp={lambda_def_opp:.3f}, λ_fav={lambda_fav:.3f}")
                        print(f"  P_MG14={P_MG14:.3f}, P_MG15={P_MG15:.3f}, P_Ge5={P_ge5:.3f}, gol_fav_reali={fav_goals}")

                # Applica tutte le configurazioni Poisson
                for name, cfg in CHECKLIST_CONFIGS.items():
                    if lambda_fav is None:
                        continue
                    if passes_config(
                        cfg, n_fav, n_opp,
                        lambda_att, lambda_def_opp, lambda_fav,
                        P_MG14, P_MG15, P_ge5
                    ):
                        sel_total[name] += 1
                        if 1 <= fav_goals <= 4:
                            sel_mg14_hits[name] += 1
                        if fav_goals >= 1:
                            sel_mg15_hits[name] += 1

        # --------------------------------------------
        # UPDATE STORICO DOPO aver usato il match
        # --------------------------------------------
        season = get_season(row)
        h_team = row["home_team"]
        a_team = row["away_team"]
        h_gf = int(row["home_ft"])
        a_gf = int(row["away_ft"])

        # home side history
        history[(h_team, season, "home")].update(h_gf, a_gf)
        # away side history
        history[(a_team, season, "away")].update(a_gf, h_gf)

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    print("\n==================== RISULTATI CHECKLIST STATISTICA MG GRID (Poisson) ====================\n")

    print(f"Totale match nel dataset: {len(df)}")
    print(f"Match nel CONTEXT MG (favorita quota 1.15–1.90): {ctx_total}")
    if ctx_total == 0:
        print("❌ Nessun match nel contesto MG. Controlla bk_p1/bk_p2 e range quote.")
        print("\n🏁 BACKTEST TERMINATO.\n")
        return

    base_mg14 = ctx_mg14_hits / ctx_total
    base_mg15 = ctx_mg15_hits / ctx_total

    print("🔹 BASELINE (solo filtro quota, nessuna checklist)\n")
    print(f"MG Favorita 1–4: {ctx_mg14_hits}/{ctx_total} = {base_mg14:.3f} ({base_mg14*100:.1f}%)")
    print(f"MG Favorita 1–5: {ctx_mg15_hits}/{ctx_total} = {base_mg15:.3f} ({base_mg15*100:.1f}%)")
    print("\n------------------------------------------------------\n")

    # Tabella riepilogo
    rows_summary = []
    for name in sorted(CHECKLIST_CONFIGS.keys()):
        n_sel = sel_total[name]
        if n_sel > 0:
            rate14 = sel_mg14_hits[name] / n_sel
            rate15 = sel_mg15_hits[name] / n_sel
            cov = n_sel / ctx_total
            uplift14 = rate14 - base_mg14
            uplift15 = rate15 - base_mg15
        else:
            rate14 = rate15 = cov = uplift14 = uplift15 = math.nan

        rows_summary.append({
            "version": name,
            "selected": n_sel,
            "coverage": cov,
            "MG14_rate": rate14,
            "MG15_rate": rate15,
            "uplift_MG14": uplift14,
            "uplift_MG15": uplift15,
        })

    summary_df = pd.DataFrame(rows_summary)

    # Stampa per versione
    for row in rows_summary:
        name = row["version"]
        n_sel = row["selected"]
        cov = row["coverage"]
        rate14 = row["MG14_rate"]
        rate15 = row["MG15_rate"]
        uplift14 = row["uplift_MG14"]
        uplift15 = row["uplift_MG15"]

        print(f"==================== {name} ====================\n")
        print(f"Match selezionati: {n_sel} (copertura: "
              f"{cov:.3f} ({cov*100:.1f}%) )" if n_sel > 0 else "Match selezionati: 0")
        if n_sel > 0:
            print(f"MG Favorita 1–4: {rate14:.3f} ({rate14*100:.1f}%)")
            print(f"MG Favorita 1–5: {rate15:.3f} ({rate15*100:.1f}%)")
            print(f"Uplift vs baseline MG 1–4: {uplift14:+.3f} ({uplift14*100:+.1f} pp)")
            print(f"Uplift vs baseline MG 1–5: {uplift15:+.3f} ({uplift15*100:+.1f} pp)")
        print("\n------------------------------------------------------\n")

    print("📊 Riepilogo tabellare:\n")
    # formattiamo un po' la stampa
    with pd.option_context('display.max_columns', None, 'display.width', 120):
        print(summary_df)

    print("\n🏁 GRID SEARCH POISSON COMPLETATA.\n")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    # se vuoi limitare i match per debug:
    # run_backtest(n_matches=20000)
    run_backtest(n_matches=None)


if __name__ == "__main__":
    main()