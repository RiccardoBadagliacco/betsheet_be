
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
from .utils import shin_derake, build_mu_priors, compute_recent_form
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
    
    home_team_id: Optional[str] = None
    away_team_id: Optional[str] = None


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
            
        self.enable_gate = False  # disattivo di default
        self.gate_tau_home = 0.65
        self.gate_tau_away = 0.60
            
        # P2: carica i priors di lega/stagione
        try:
            self.mu_priors = build_mu_priors()
            print(f"[P2] Loaded {len(self.mu_priors)} μ_total priors from DB.")
            # 👇 AGGIUNGI QUESTO BLOCCO
        except Exception as e:
            print(f"[WARN] Could not load μ priors: {e}")
            self.mu_priors = {}
            
        try:

            self.form_df = compute_recent_form()
            print(self.form_df.sample(5))
            print(self.form_df["gf_recent"].describe())
            print(f"[P1] Loaded recent form table: {len(self.form_df)} entries")
        except Exception as e:
            print(f"[WARN] Could not load recent form data: {e}")
            self.form_df = None

    # ---------- Public API ----------
    
    def _derake_shin(self, odds):
        """
        Usa il metodo di Shin (1993) per derakare le quote 1X2.
        """
        try:
            if any(o is None or o <= 1.0 for o in odds):
                return self._derake_threeway(*odds)   # derake proporzionale di fallback
            probs, zeta = shin_derake(odds)
            return probs
        except Exception:
            return self._derake_threeway(*odds)

    def predict(self, data: Dict) -> MGOutput:
        """
        data: dict compatible with MGInput fields
        Returns: MGOutput with probabilities for the six MG markets.
        """
        inp = MGInput(**data)
        
        self.current_league = inp.league_code
        self.current_season = inp.season
        
        self.home_id = inp.home_team_id
        self.away_id = inp.away_team_id
                
        # P2: normalizza il formato season (es. "2025_2026" -> "2025/2026")
        if inp.season and "_" in inp.season:
            inp.season = inp.season.replace("_", "/")

        # 1) Derake 1X2 (and O/U if present)
        p1, px, p2 = self._derake_shin([
            inp.avg_home_odds, inp.avg_draw_odds, inp.avg_away_odds
        ])

        p_over25 = p_under25 = None
        p_over25, p_under25, _ = self._derake_threeway(
            inp.avg_over_25_odds, inp.avg_under_25_odds, None, two_way=True
        )

        # 2) Estimate lambdas
        lam_h, lam_a = self._estimate_lambdas(
            p1=p1, px=px, p2=p2,
            p_over25=p_over25,
            mu_total_prior=inp.mu_total_prior,
            league_code=inp.league_code,
            season=inp.season,
        )

        # 3) Raw MG probs from Poisson marginals
        raw = self._compute_mg_probs(lam_h, lam_a)
        
        if inp.avg_home_odds and inp.avg_home_odds > self.home_fav_threshold:
            for k in ["MG_HOME_1_3","MG_HOME_1_4","MG_HOME_1_5"]:
                raw[k] = None
        if inp.avg_away_odds and inp.avg_away_odds > self.away_fav_threshold:
            for k in ["MG_AWAY_1_3","MG_AWAY_1_4","MG_AWAY_1_5"]:
                raw[k] = None
        
        if hasattr(self, "enable_gate") and self.enable_gate:
            # Probabilità che ciascuna segni almeno 1
            p_h_ge1 = 1.0 - math.exp(-lam_h)
            p_a_ge1 = 1.0 - math.exp(-lam_a)

            # Favorito dal derake 1X2
            # (usa p1, p2 già calcolate: home favorito se p1 > p2)
            is_home_fav = (p1 is not None and p2 is not None and p1 > p2)
            is_away_fav = (p1 is not None and p2 is not None and p2 > p1)

            # soglie (parametrizzabili)
            tau_home = getattr(self, "gate_tau_home", 0.65)  # default
            tau_away = getattr(self, "gate_tau_away", 0.60)  # default

            # se favorito non supera la soglia, azzera i mercati del suo lato
            if is_home_fav and (p_h_ge1 < tau_home):
                for k in ("MG_HOME_1_3", "MG_HOME_1_4", "MG_HOME_1_5"):
                    raw[k] = None
            if is_away_fav and (p_a_ge1 < tau_away):
                for k in ("MG_AWAY_1_3", "MG_AWAY_1_4", "MG_AWAY_1_5"):
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
        league_code: Optional[str] = None,
        season: Optional[str] = None,
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
        elif hasattr(self, "mu_priors"):
            key = (league_code, season)
            mu = self.mu_priors.get(key, 2.6)
        else:
            mu = float(mu_total_prior) if mu_total_prior else 2.6
            
        # μ adattivo (baseline P3.2)
        if p1 is not None and p2 is not None:
            imbalance = abs(p1 - p2)
            if imbalance > 0.2:
                alpha = 0.18
                mu *= (1 - alpha * imbalance)

        # Step B: find delta (difference in means) consistent with 1X2
        # Bounds for delta: must be within [-mu+eps, mu-eps], but allow a bit wider for safety
        delta = self._solve_delta_from_1x2(mu, p1, px, p2)

        lam_h = max(0.05, (mu + delta) / 2.0)
        lam_a = max(0.05, (mu - delta) / 2.0)
        
        if getattr(self, "form_df", None) is not None:
            try:
                form = self.form_df
                league = league_code
                season = season

                # Media dei gol totali di lega per normalizzare
                mu_league = np.mean([
                    v for k, v in self.mu_priors.items()
                    if k[0] == league
                ]) if self.mu_priors else 2.6
                mu_home_league = mu_league / 2.0

                # Recupera forma recente di home e away (EMA smussata)
                gf_home = form.loc[
                    (form["team_id"] == getattr(self, "home_id", None)) &
                    (form["season"] == season),
                    "gf_recent"
                ].mean()
                ga_home = form.loc[
                    (form["team_id"] == getattr(self, "home_id", None)) &
                    (form["season"] == season),
                    "ga_recent"
                ].mean()
                gf_away = form.loc[
                    (form["team_id"] == getattr(self, "away_id", None)) &
                    (form["season"] == season),
                    "gf_recent"
                ].mean()
                ga_away = form.loc[
                    (form["team_id"] == getattr(self, "away_id", None)) &
                    (form["season"] == season),
                    "ga_recent"
                ].mean()

                # --- (1) Correzione individuale di forma ---
                alpha = 0.3
                if not np.isnan(gf_home) and mu_home_league > 0:
                    old_h = lam_h
                    ratio = gf_home / mu_home_league
                    lam_h *= (1 + alpha * (ratio - 1))
                    print(f"[Form] Home ratio={ratio:.2f} λ_h: {old_h:.2f}→{lam_h:.2f}")

                if not np.isnan(ga_away) and mu_home_league > 0:
                    old_a = lam_a
                    ratio = ga_away / mu_home_league
                    lam_a *= (1 + alpha * (ratio - 1))
                    print(f"[Form] Away ratio={ratio:.2f} λ_a: {old_a:.2f}→{lam_a:.2f}")

                # --- (2) MATCHUP ADJUSTMENT ---
                if not np.isnan(gf_home) and not np.isnan(ga_away):
                    old_h = lam_h
                    adj = 0.5 * (gf_home / mu_home_league) + 0.5 * (ga_away / mu_home_league)
                    lam_h *= (1 + 0.25 * (adj - 1))
                    print(f"[Form-MU] Home matchup adj: {adj:.2f} λ_h {old_h:.2f}→{lam_h:.2f}")

                if not np.isnan(ga_home) and not np.isnan(gf_away):
                    old_a = lam_a
                    adj = 0.5 * (gf_away / mu_home_league) + 0.5 * (ga_home / mu_home_league)
                    lam_a *= (1 + 0.25 * (adj - 1))
                    print(f"[Form-MU] Away matchup adj: {adj:.2f} λ_a {old_a:.2f}→{lam_a:.2f}")

            except Exception as e:
                print(f"[Form][WARN] adjustment failed: {e}")

        # --- (3) Clipping finale di sicurezza ---
        lam_h = np.clip(lam_h, 0.4, 3.5)
        lam_a = np.clip(lam_a, 0.3, 3.0)
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
