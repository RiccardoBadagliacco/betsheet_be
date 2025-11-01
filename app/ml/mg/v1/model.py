
# app/ml/mg_model/v1/model.py
# Baseline MGModelV1 — probabilistic/statistical, no heavy ML
# - Derake 1X2 (+ O/U2.5 if available)
# - Estimate (lambda_home, lambda_away) from 1X2 (+ O/U2.5 or league prior)
# - Compute MG probabilities: Home/ Away in [1..3], [1..4], [1..5]
# - Optional isotonic calibration per-league with global fallback
#
# This file is designed to mirror the style of OverModelV1 but tailored to MG markets.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    # Optional: used if calibrators are present (sklearn IsotonicRegression pickles)
    import pickle
except Exception:
    pickle = None


@dataclass
class MGInput:
    avg_home_odds: float
    avg_draw_odds: float
    avg_away_odds: float
    avg_over_25_odds: Optional[float] = None
    avg_under_25_odds: Optional[float] = None
    league_code: Optional[str] = None
    season: Optional[str] = None  # free text (e.g., "2023/2024")
    # Optional prior for total goals if O/U is missing
    mu_total_prior: Optional[float] = None  # typical ~2.6 (league-season prior)


@dataclass
class MGOutput:
    probs: Dict[str, float]
    lambdas: Dict[str, float]
    calibrated: bool
    meta: Dict[str, str]


MARKETS = (
    "MG_HOME_1_3",
    "MG_HOME_1_4",
    "MG_HOME_1_5",
    "MG_AWAY_1_3",
    "MG_AWAY_1_4",
    "MG_AWAY_1_5",
)


class MGModelV1:
    """
    Baseline model for MG markets (Home/Away goals in ranges 1–3, 1–4, 1–5).
    - Independent Poisson assumption for home/away goals.
    - Inference of (lambda_home, lambda_away) from 1X2 and optional O/U 2.5.
    - If O/U missing, fallback to league-season prior for total goals (mu_total_prior).
    - Optional isotonic calibration per-market per-league with global fallback.
    """

    def __init__(
        self,
        patch: int = 0, 
        calibrators_global_dir: Optional[str] = None,
        calibrators_leagues_dir: Optional[str] = None,
        use_league_calibrators: bool = True,
        home_fav_threshold: float = 1.8,
        away_fav_threshold: float = 1.9,
        max_goals_grid: int = 12,
    ):
        self.patch = patch
        self.version = f"MGModelV1-P{patch}"
        self.home_fav_threshold = home_fav_threshold
        self.away_fav_threshold = away_fav_threshold
        self.max_goals_grid = int(max_goals_grid)
        self.use_league_calibrators = bool(use_league_calibrators)

        self.cal_global = {}
        self.cal_leagues = {}
        if calibrators_global_dir and pickle is not None:
            self.cal_global = self._load_calibrators_dir(calibrators_global_dir)
        if calibrators_leagues_dir and pickle is not None and self.use_league_calibrators:
            self.cal_leagues = self._load_calibrators_dir(calibrators_leagues_dir, by_league=True)

    # ---------- Public API ----------

    def predict(self, data: Dict) -> MGOutput:
        """
        data: dict compatible with MGInput fields
        Returns: MGOutput with probabilities for the six MG markets.
        """
        inp = MGInput(**data)

        # 1) Derake 1X2 (and O/U if present)
        p1, px, p2 = self._derake_threeway(
            inp.avg_home_odds, inp.avg_draw_odds, inp.avg_away_odds
        )

        p_over25 = p_under25 = None
        p_over25, p_under25, _ = self._derake_threeway(
            inp.avg_over_25_odds, inp.avg_under_25_odds, None, two_way=True
        )

        # 2) Estimate lambdas
        lam_h, lam_a = self._estimate_lambdas(
            p1=p1, px=px, p2=p2, p_over25=p_over25, mu_total_prior=inp.mu_total_prior
        )

        # 3) Raw MG probs from Poisson marginals
        raw = self._compute_mg_probs(lam_h, lam_a)
        
        # --- PATCH 1: MG selettivo per favoriti ---
        if inp.avg_home_odds and inp.avg_home_odds > self.home_fav_threshold:
            for k in ["MG_HOME_1_3","MG_HOME_1_4","MG_HOME_1_5"]:
                raw[k] = None
        if inp.avg_away_odds and inp.avg_away_odds > self.away_fav_threshold:
            for k in ["MG_AWAY_1_3","MG_AWAY_1_4","MG_AWAY_1_5"]:
                raw[k] = None

        # 4) Calibration (league-specific -> global fallback)
        calibrated = False
        if self.cal_global or self.cal_leagues:
            raw = self._apply_calibration(raw, league_code=inp.league_code)
            calibrated = True

        meta = {
            "version": self.version,
            "has_ou25": str(p_over25 is not None),
            "league_code": inp.league_code or "",
            "season": inp.season or "",
        }

        return MGOutput(
            probs=raw,
            lambdas={"home": float(lam_h), "away": float(lam_a)},
            calibrated=calibrated,
            meta=meta,
        )

    # ---------- Core computations ----------

    @staticmethod
    def _derake_threeway(
        odd_a: float, odd_b: float, odd_c: Optional[float], two_way: bool = False
    ) -> Tuple[float, float, Optional[float]]:
        """
        Proportional deraking for 2-way or 3-way markets.
        Returns probabilities (a,b,c) if 3-way, or (a,b,None) if 2-way.
        """
        def inv(o):
            return 0.0 if (o is None or o <= 0) else (1.0 / o)

        if two_way:
            pa, pb = inv(odd_a), inv(odd_b)
            s = pa + pb
            if s <= 0:
                return 0.5, 0.5, None
            return pa / s, pb / s, None

        pa, pb, pc = inv(odd_a), inv(odd_b), inv(odd_c)
        s = pa + pb + pc
        if s <= 0:
            return (1/3), (1/3), (1/3)
        return pa / s, pb / s, pc / s

    def _estimate_lambdas(
        self,
        p1: float,
        px: float,
        p2: float,
        p_over25: Optional[float] = None,
        mu_total_prior: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute (lambda_home, lambda_away).
        - If O/U present: infer mu_total from P(Total>=3) and asymmetry from 1X2.
        - Else: use mu_total_prior (fallback 2.6 if None).
        Asymmetry is found by matching P(HomeWin) from Poisson grid to p1 (weighted also with px, p2).
        """
        # Step A: total goals (mu)
        if p_over25 is not None:
            mu = self._invert_over25_to_mu(p_over25)
        else:
            mu = float(mu_total_prior) if (mu_total_prior and mu_total_prior > 0) else 2.6

        # Step B: find delta (difference in means) consistent with 1X2
        # Bounds for delta: must be within [-mu+eps, mu-eps], but allow a bit wider for safety
        delta = self._solve_delta_from_1x2(mu, p1, px, p2)

        lam_h = max(0.05, (mu + delta) / 2.0)
        lam_a = max(0.05, (mu - delta) / 2.0)
        return lam_h, lam_a

    def _invert_over25_to_mu(self, p_over25: float) -> float:
        """
        Invert P(Total >= 3) under Poisson(total with mean mu) to find mu.
        P(T>=3) = 1 - [e^{-mu} * (1 + mu + mu^2/2)]
        Use bisection over [0.2, 6.0].
        """
        p_over25 = float(np.clip(p_over25, 1e-6, 1 - 1e-6))

        def p_ge3(mu):
            # 1 - P(0) - P(1) - P(2)
            e = math.exp(-mu)
            return 1.0 - e * (1.0 + mu + (mu * mu) / 2.0)

        lo, hi = 0.2, 6.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if p_ge3(mid) < p_over25:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _solve_delta_from_1x2(self, mu: float, p1: float, px: float, p2: float) -> float:
        """
        Given total mean mu and target 1X2 probs, find delta such that Poisson grid implied 1X2
        (with lambdas = (mu±delta)/2) matches the target. Minimize squared error over p1,px,p2.
        Search delta in a bounded interval.
        """
        target = np.array([p1, px, p2], dtype=float)

        def error_for_delta(delta: float) -> float:
            lam_h = max(0.01, (mu + delta) / 2.0)
            lam_a = max(0.01, (mu - delta) / 2.0)
            p1g, pxg, p2g = self._oneXtwo_from_lambdas(lam_h, lam_a)
            pred = np.array([p1g, pxg, p2g], dtype=float)
            return float(np.sum((pred - target) ** 2))

        # Bracket delta: reasonable range [-3, +3], refined
        lo, hi = -3.0, 3.0
        # Golden-section search
        phi = (1 + 5 ** 0.5) / 2.0
        invphi = 1 / phi
        a, b = lo, hi
        c = b - invphi * (b - a)
        d = a + invphi * (b - a)
        fc = error_for_delta(c)
        fd = error_for_delta(d)
        for _ in range(80):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - invphi * (b - a)
                fc = error_for_delta(c)
            else:
                a, c, fc = c, d, fd
                d = a + invphi * (b - a)
                fd = error_for_delta(d)
        delta_opt = (a + b) / 2.0
        return float(delta_opt)

    def _oneXtwo_from_lambdas(self, lam_h: float, lam_a: float) -> Tuple[float, float, float]:
        """
        Compute 1X2 probabilities from (lam_h, lam_a) via truncated grid sums.
        """
        K = self.max_goals_grid
        # Precompute pmfs
        ph = np.array([self._poisson_pmf(k, lam_h) for k in range(K + 1)])
        pa = np.array([self._poisson_pmf(k, lam_a) for k in range(K + 1)])

        # Matrix of joint probs (independent)
        joint = np.outer(ph, pa)  # shape (K+1, K+1)

        p_home = float(np.tril(joint, -1).sum())  # h > a
        p_draw = float(np.trace(joint))           # h = a
        p_away = float(np.triu(joint, +1).sum())  # a > h

        # Tiny mass beyond K is ignored; we renormalize slightly to 1
        s = p_home + p_draw + p_away
        if s > 0:
            p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
        return p_home, p_draw, p_away

    def _compute_mg_probs(self, lam_h: float, lam_a: float) -> Dict[str, float]:
        """
        Compute the six MG probabilities from Poisson marginals.
        """
        def range_prob(lam: float, lo: int, hi: int) -> float:
            return float(sum(self._poisson_pmf(k, lam) for k in range(lo, hi + 1)))

        out = {
            "MG_HOME_1_3": range_prob(lam_h, 1, 3),
            "MG_HOME_1_4": range_prob(lam_h, 1, 4),
            "MG_HOME_1_5": range_prob(lam_h, 1, 5),
            "MG_AWAY_1_3": range_prob(lam_a, 1, 3),
            "MG_AWAY_1_4": range_prob(lam_a, 1, 4),
            "MG_AWAY_1_5": range_prob(lam_a, 1, 5),
        }
        # Clip to [0,1] and ensure numerical sanity
        for k, v in out.items():
            out[k] = float(np.clip(v, 0.0, 1.0))
        return out

    @staticmethod
    def _poisson_pmf(k: int, lam: float) -> float:
        if k < 0:
            return 0.0
        return float(math.exp(-lam) * (lam ** k) / math.factorial(k))

    # ---------- Calibration ----------

    def _apply_calibration(self, probs: Dict[str, float], league_code: Optional[str]) -> Dict[str, float]:
        out = dict(probs)
        def safe_predict(calibrator, value):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return value
            return float(np.clip(calibrator.predict([value])[0], 0.0, 1.0))

        # Try league-specific first
        if self.use_league_calibrators and league_code and league_code in self.cal_leagues:
            calmap = self.cal_leagues[league_code]
            for mkt in MARKETS:
                if mkt in calmap:
                    out[mkt] = safe_predict(calmap[mkt], probs[mkt])
            return out

        # Global fallback
        for mkt in MARKETS:
            if mkt in self.cal_global:
                out[mkt] = safe_predict(self.cal_global[mkt], probs[mkt])
        return out

    def _load_calibrators_dir(self, path: str, by_league: bool = False):
        """
        Load pickled IsotonicRegression calibrators.
        - If by_league=False: expect files named like 'MG_HOME_1_3.pkl', etc.
        - If by_league=True: expect subfolders per league_code each with those files.
        """
        store = {}
        if pickle is None:
            return store

        import os

        if not os.path.isdir(path):
            return store

        if not by_league:
            for mkt in MARKETS:
                fp = os.path.join(path, f"{mkt}.pkl")
                if os.path.isfile(fp):
                    try:
                        with open(fp, "rb") as f:
                            store[mkt] = pickle.load(f)
                    except Exception:
                        pass
            return store

        # by_league=True
        for league_code in os.listdir(path):
            ldir = os.path.join(path, league_code)
            if not os.path.isdir(ldir):
                continue
            m = {}
            for mkt in MARKETS:
                fp = os.path.join(ldir, f"{mkt}.pkl")
                if os.path.isfile(fp):
                    try:
                        with open(fp, "rb") as f:
                            m[mkt] = pickle.load(f)
                    except Exception:
                        pass
            if m:
                store[league_code] = m
        return store
