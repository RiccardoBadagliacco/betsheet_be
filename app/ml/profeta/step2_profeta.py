# ============================================================
# app/ml/profeta/step2_profeta.py
# ============================================================

"""
STEP2 — PROFETA ENGINE (prediction)

Funzionalità:
  - carica i parametri di Profeta V0
  - per un match (storico o fixture) calcola:
        * lambda_home, lambda_away
        * matrice P(GH=i, GA=j)
        * probabilità mercati:
            - 1 X 2
            - GG / NoGG
            - Over/Under 1.5, 2.5, 3.5
            - Multigol totali: 1-3, 1-4, 1-5, 2-4, 2-5, 2-6
            - Multigol Casa: 1-3, 1-4, 1-5
            - Multigol Ospite: 1-3, 1-4, 1-5
            - 6 risultati esatti più probabili
        * altre info utili (xG, P(0-0), clean sheets, ecc.)

Input file:
    data/step0_profeta.parquet
    data/step1_profeta_params.pth
    data/step1_profeta_metadata.json
"""

import sys
from pathlib import Path
import json
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import torch
import math

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR / "data"
STEP0_PATH = DATA_DIR / "step0_profeta.parquet"
PARAMS_PATH = DATA_DIR / "step1_profeta_params.pth"
META_PATH   = DATA_DIR / "step1_profeta_metadata.json"

DC_PARAMS_PATH = DATA_DIR / "step1b_profeta_dc_params.json"
try:
    with open(DC_PARAMS_PATH, "r") as f:
        DC_PARAMS = json.load(f)
    RHO_DC = float(DC_PARAMS.get("rho_dc", 0.0))
except FileNotFoundError:
    DC_PARAMS = {}
    RHO_DC = 0.0

print(f"🔧 Dixon–Coles attivo: rho_dc = {RHO_DC:.4f}")
# ============================================================
# UTILS
# ============================================================

def poisson_pmf_vector(lmbda: float, max_goals: int) -> np.ndarray:
    """
    Calcola il vettore P(G = k) per k=0..max_goals per una Poisson(lmbda).
    Rinomralizza leggermente per far sommare a 1 (la coda > max_goals è piccola).
    """
    ks = np.arange(0, max_goals + 1)
    pmf = np.exp(-lmbda) * np.power(lmbda, ks) / np.array([math.factorial(k) for k in ks], dtype=float)
    s = pmf.sum()
    if s > 0:
        pmf /= s
    return pmf


def build_score_matrix(lmbda_home: float, lmbda_away: float, max_goals: int) -> np.ndarray:
    """
    Costruisce la matrice P(GH=i, GA=j) assumendo Poisson indipendenti.
    shape = (max_goals+1, max_goals+1)
    """
    ph = poisson_pmf_vector(lmbda_home, max_goals)  # (G+1,)
    pa = poisson_pmf_vector(lmbda_away, max_goals)
    # prodotto esterno
    return np.outer(ph, pa)  # [i,j] = P(GH=i)*P(GA=j)

def apply_dixon_coles_matrix(P, lam_h, lam_a, rho):
    """
    Applica la correzione Dixon–Coles ai 4 casi:
    (0,0), (1,0), (0,1), (1,1).
    """
    P = P.copy()

    def C(x, y):
        if x == 0 and y == 0:
            return 1.0 + rho * (-lam_h * lam_a)
        elif x == 0 and y == 1:
            return 1.0 + rho * lam_h
        elif x == 1 and y == 0:
            return 1.0 + rho * lam_a
        elif x == 1 and y == 1:
            return 1.0 + rho * (1 - lam_h - lam_a)
        else:
            return 1.0

    # solo i 4 casi speciali
    for x in [0,1]:
        for y in [0,1]:
            factor = C(x, y)
            if factor > 0:
                P[x, y] *= factor

    # rinormalizza
    s = P.sum()
    if s > 0:
        P /= s

    return P

def apply_tail_heavy_matrix(
    P: np.ndarray,
    lam_h: float,
    lam_a: float,
    eps: float = 0.02,
    infl: float = 1.20,
    max_goals: int = 10,
) -> np.ndarray:
    """
    Mixture:
      (1 - eps) * P_original
      eps       * P_poisson_inflated
    """
    # Poisson con lambda gonfiati
    P_tail = build_score_matrix(
        lam_h * infl,
        lam_a * infl,
        max_goals,
    )

    P_mix = (1.0 - eps) * P + eps * P_tail

    # rinormalizza
    s = P_mix.sum()
    if s > 0:
        P_mix /= s

    return P_mix

# ============================================================
# PROFETA ENGINE
# ============================================================

class ProfetaEngine:
    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals

        # Carica metadata
        with open(META_PATH, "r") as f:
            meta = json.load(f)

        self.league_to_idx: Dict[str, int] = meta["league_to_idx"]
        self.season_to_idx: Dict[str, int] = meta["season_to_idx"]
        self.ts_to_idx: Dict[str, int]     = meta["teamseason_to_idx"]

        # Carica parametri
        state = torch.load(PARAMS_PATH, map_location=torch.device("cpu"))

        # Tensors
        self.mu = state["mu"].item()
        self.gamma_league = state["gamma_league"].numpy()
        self.hfa_league   = state["hfa_league"].numpy()
        self.delta_season = state["delta_season"].numpy()
        self.att          = state["att"].numpy()
        self.defn         = state["defn"].numpy()

        self.feature_home = meta["feature_home"]
        self.feature_away = meta["feature_away"]

        self.beta_home = state["beta_home"].numpy()
        self.beta_away = state["beta_away"].numpy()

    # --------------------------------------------------------
    # LAMBDA HOME/AWAY PER UN MATCH
    # --------------------------------------------------------
    def compute_lambdas(
        self,
        league_id: str,
        season_id: str,
        home_team_season_id: str,
        away_team_season_id: str,
        X_home: np.ndarray,
        X_away: np.ndarray,
    ) -> Tuple[float, float]:

        l_idx = self.league_to_idx[league_id]
        s_idx = self.season_to_idx[season_id]

        if l_idx is None or s_idx is None:
            # fallback globale
            l_idx = 0
            s_idx = 0

        h_ts_idx = self.ts_to_idx.get(home_team_season_id)
        a_ts_idx = self.ts_to_idx.get(away_team_season_id)

        att_h = self.att[h_ts_idx] if h_ts_idx is not None else 0.0
        def_h = self.defn[h_ts_idx] if h_ts_idx is not None else 0.0
        att_a = self.att[a_ts_idx] if a_ts_idx is not None else 0.0
        def_a = self.defn[a_ts_idx] if a_ts_idx is not None else 0.0

        # 👉 FEATURE LINEAR TERMS (MANCAVANO)
        lin_home = float(np.dot(self.beta_home, X_home))
        lin_away = float(np.dot(self.beta_away, X_away))

        log_lambda_home = (
            self.mu
            + self.gamma_league[l_idx]
            + self.delta_season[s_idx]
            + self.hfa_league[l_idx]
            + att_h
            - def_a
            + lin_home
        )

        log_lambda_away = (
            self.mu
            + self.gamma_league[l_idx]
            + self.delta_season[s_idx]
            + att_a
            - def_h
            + lin_away
        )

        log_lambda_home = np.clip(log_lambda_home, -10, 10)
        log_lambda_away = np.clip(log_lambda_away, -10, 10)

        return float(np.exp(log_lambda_home)), float(np.exp(log_lambda_away))

    # --------------------------------------------------------
    # MERCATI DAL SCORE MATRIX
    # --------------------------------------------------------
    def markets_from_score_matrix(self, M: np.ndarray) -> Dict[str, Any]:
        """
        M: matrice P(GH=i, GA=j)
        Ritorna dizionario con tutti i mercati richiesti + extra.
        """
        G = self.max_goals
        probs = {}

        # helper lambda
        def sum_cond(cond):
            total = 0.0
            for i in range(G + 1):
                for j in range(G + 1):
                    if cond(i, j):
                        total += M[i, j]
            return float(total)

        # 1 X 2
        probs["p1"] = sum_cond(lambda i, j: i > j)
        probs["px"] = sum_cond(lambda i, j: i == j)
        probs["p2"] = sum_cond(lambda i, j: i < j)

        # GG / NOGG
        probs["p_gg"]    = sum_cond(lambda i, j: i >= 1 and j >= 1)
        probs["p_nogg"]  = 1.0 - probs["p_gg"]

        # Over/Under totali
        def total_goals_cond(min_g=None, max_g=None):
            return sum_cond(
                lambda i, j: (min_g is None or i + j >= min_g)
                             and (max_g is None or i + j <= max_g)
            )

        probs["p_over_1_5"] = total_goals_cond(min_g=2)
        probs["p_under_1_5"] = 1.0 - probs["p_over_1_5"]

        probs["p_over_2_5"] = total_goals_cond(min_g=3)
        probs["p_under_2_5"] = 1.0 - probs["p_over_2_5"]

        probs["p_over_3_5"] = total_goals_cond(min_g=4)
        probs["p_under_3_5"] = 1.0 - probs["p_over_3_5"]

        # Multigol totali
        def mg_total(a, b):
            return total_goals_cond(min_g=a, max_g=b)

        probs["p_mg_1_3"] = mg_total(1, 3)
        probs["p_mg_1_4"] = mg_total(1, 4)
        probs["p_mg_1_5"] = mg_total(1, 5)
        probs["p_mg_2_4"] = mg_total(2, 4)
        probs["p_mg_2_5"] = mg_total(2, 5)
        probs["p_mg_2_6"] = mg_total(2, 6)

        # Distribuzioni marginali casa/ospite
        ph = M.sum(axis=1)  # P(GH=i)
        pa = M.sum(axis=0)  # P(GA=j)

        def mg_team(ph_or_pa: np.ndarray, a: int, b: int) -> float:
            a = max(a, 0)
            b = min(b, self.max_goals)
            if a > b:
                return 0.0
            return float(ph_or_pa[a:b+1].sum())

        # MG Casa
        probs["p_mg_home_1_3"] = mg_team(ph, 1, 3)
        probs["p_mg_home_1_4"] = mg_team(ph, 1, 4)
        probs["p_mg_home_1_5"] = mg_team(ph, 1, 5)

        # MG Ospite
        probs["p_mg_away_1_3"] = mg_team(pa, 1, 3)
        probs["p_mg_away_1_4"] = mg_team(pa, 1, 4)
        probs["p_mg_away_1_5"] = mg_team(pa, 1, 5)

        # 6 risultati esatti più probabili
        score_probs: List[Tuple[str, float]] = []
        for i in range(G + 1):
            for j in range(G + 1):
                score_probs.append((f"{i}-{j}", float(M[i, j])))

        score_probs.sort(key=lambda x: x[1], reverse=True)
        probs["top6_correct_score"] = score_probs[:6]

        # Extra utili
        probs["p_0_0"] = float(M[0, 0])
        probs["p_home_clean_sheet"] = sum_cond(lambda i, j: j == 0)
        probs["p_away_clean_sheet"] = sum_cond(lambda i, j: i == 0)

        return probs

    # --------------------------------------------------------
    # PREDICT PER UN MATCH (ROW DI step0_profeta.parquet)
    # --------------------------------------------------------
    def predict_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        row: riga del dataframe step0_profeta
        Ritorna:
            {
              "lambda_home": ...,
              "lambda_away": ...,
              "goal_matrix": M (np.ndarray),
              "markets": { ... }
            }
        """
        league_id = str(row["league_id"])
        season_id = str(row["season_id"])
        home_ts   = str(row["home_team_season_id"])
        away_ts   = str(row["away_team_season_id"])

        X_home = np.array([row.get(c, 0.0) for c in self.feature_home], dtype=float)
        X_away = np.array([row.get(c, 0.0) for c in self.feature_away], dtype=float)

        X_home = np.nan_to_num(X_home)
        X_away = np.nan_to_num(X_away)

        lam_h, lam_a = self.compute_lambdas(
            league_id, season_id, home_ts, away_ts, X_home, X_away
        )
        M = build_score_matrix(lam_h, lam_a, self.max_goals)

        # 👉 Correzione Dixon–Coles sui 4 casi (0,0), (1,0), (0,1), (1,1)
        if abs(RHO_DC) > 1e-6:
            M = apply_dixon_coles_matrix(M, lam_h, lam_a, RHO_DC)

        # Tail-heavy (score alti)
        """ M = apply_tail_heavy_matrix(
            M,
            lam_h,
            lam_a,
            eps=0.04,
            infl=1.35,
            max_goals=self.max_goals,
        ) """

        markets = self.markets_from_score_matrix(M)

        # Aggiungiamo xG totali
        markets["lambda_home"] = lam_h
        markets["lambda_away"] = lam_a
        markets["xg_total"] = lam_h + lam_a

        return {
            "lambda_home": lam_h,
            "lambda_away": lam_a,
            "goal_matrix": M,
            "markets": markets,
        }


# ============================================================
# DEMO / MAIN
# ============================================================

def main():
    print("🚀 PROFETA STEP2 — Demo prediction")

    df = pd.read_parquet(STEP0_PATH)

    # Esempio: prendo la prima fixture (se c'è), altrimenti il primo match storico
    fixtures = df[df["is_fixture"] == True]
    if len(fixtures) > 0:
        row = fixtures.iloc[0]
        print("📌 Uso la prima FIXTURE come esempio.")
    else:
        row = df[df["is_fixture"] == False].iloc[0]
        print("📌 Nessuna fixture trovata, uso il primo match storico.")

    engine = ProfetaEngine(max_goals=10)
    result = engine.predict_from_row(row)

    lam_h = result["lambda_home"]
    lam_a = result["lambda_away"]
    markets = result["markets"]
    M = result["goal_matrix"]
    markets["p_high_scoring"] = float(np.sum(M[4:, 4:]))

    print(f"\n⚙️ Lambda home = {lam_h:.3f}, lambda away = {lam_a:.3f}")
    print(f"   xG totale   = {markets['xg_total']:.3f}")

    print("\n🎯 Probabilità mercati principali:")
    print(f"  1  = {markets['p1']:.3%}")
    print(f"  X  = {markets['px']:.3%}")
    print(f"  2  = {markets['p2']:.3%}")
    print(f"  GG   = {markets['p_gg']:.3%}")
    print(f"  NoGG = {markets['p_nogg']:.3%}")
    print(f"  Over 2.5 = {markets['p_over_2_5']:.3%}")
    print(f"  Under2.5 = {markets['p_under_2_5']:.3%}")

    print("\n📦 Multigol totali:")
    for k in ["p_mg_1_3", "p_mg_1_4", "p_mg_1_5", "p_mg_2_4", "p_mg_2_5", "p_mg_2_6"]:
        print(f"  {k} = {markets[k]:.3%}")

    print("\n📦 Multigol CASA:")
    for k in ["p_mg_home_1_3", "p_mg_home_1_4", "p_mg_home_1_5"]:
        print(f"  {k} = {markets[k]:.3%}")

    print("\n📦 Multigol OSPITE:")
    for k in ["p_mg_away_1_3", "p_mg_away_1_4", "p_mg_away_1_5"]:
        print(f"  {k} = {markets[k]:.3%}")

    print("\n🎯 Top 6 risultati esatti più probabili:")
    for score, p in markets["top6_correct_score"]:
        print(f"  {score}: {p:.3%}")

    print("\n✅ PROFETA STEP2 demo completata.")


if __name__ == "__main__":
    main()
