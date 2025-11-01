# app/ml/over_model/v1/model.py
from __future__ import annotations
from typing import Dict, Any, Optional
import os, math, pickle

CALIB_BASE = os.path.join(os.path.dirname(__file__), "calibrators")
CALIB_GLOBAL = os.path.join(CALIB_BASE, "global")
CALIB_LEAGUES = os.path.join(CALIB_BASE, "leagues")

# === Config soglie operative (dalla tua sweep) ===
DEFAULT_THRESHOLDS = {
    "O0_5": 0.90,  # banker
    "O1_5": 0.85,  # high volume
    "O2_5": 0.65,  # premium select
}


def _solve_lambda_from_pover25(p_over: float) -> float:
    # inverte P(T>2.5)=p via Poisson; robusto a [0.01, 6.0]
    lo, hi = 0.01, 6.0
    for _ in range(60):
        mid = (lo+hi)/2.0
        p0 = math.exp(-mid)
        p1 = p0*mid
        p2 = p1*(mid/2.0)
        pover = 1.0 - (p0+p1+p2)
        if pover < p_over: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def _poisson_over_probs(lam: float):
    p0 = math.exp(-lam)
    p1 = p0*lam
    p2 = p1*(lam/2.0)
    return {
        "p05": 1.0 - p0,
        "p15": 1.0 - (p0+p1),
        "p25": 1.0 - (p0+p1+p2),
    }

class OverModelV1:
    """
    Baseline nuova (ex Patch 7):
      - input: sole quote O/U 2.5 (Avg>2.5, Avg<2.5) e opzionale league_code
      - step 1: rimuove vig (fair p_over_2.5)
      - step 2: risolve λ dal solo O2.5 (Poisson)
      - step 3: calcola p(>0.5,>1.5,>2.5) via Poisson
      - step 4: calibra con Isotonic (league-specific se disponibile; altrimenti global)
      - output: p05, p15, p25 calibrate + EV O2.5
    """

    def __init__(self, db_path: Optional[str] = None, debug: bool = False):
        self.debug = debug
        # carica calibratori globali all'avvio
        self._iso_g = {}
        for suf in ("05", "15", "25"):
            path = os.path.join(CALIB_GLOBAL, f"iso_poisson_{suf}.pkl")
            try:
                with open(path, "rb") as f:
                    self._iso_g[suf] = pickle.load(f)
            except Exception as e:
                self._iso_g[suf] = None
                if self.debug: print(f"⚠️ missing global calibrator {path}: {e}")

        # cache per leghe
        self._iso_league_cache = {}

    @staticmethod
    def _fair_over_prob(over: float, under: float) -> Optional[float]:
        if over is None or under is None: return None
        try:
            over = float(over); under = float(under)
            if min(over, under) <= 1.01: return None
            inv = 1.0/over + 1.0/under
            return (1.0/over) / inv
        except Exception:
            return None

    def _load_league_calibs(self, league_code: str):
        if league_code in self._iso_league_cache:
            return self._iso_league_cache[league_code]
        paths = {
            "05": os.path.join(CALIB_LEAGUES, f"{league_code}_05.pkl"),
            "15": os.path.join(CALIB_LEAGUES, f"{league_code}_15.pkl"),
            "25": os.path.join(CALIB_LEAGUES, f"{league_code}_25.pkl"),
        }
        iso = {}
        try:
            for k, p in paths.items():
                with open(p, "rb") as f:
                    iso[k] = pickle.load(f)
            if self.debug: print(f"✅ loaded league calibrators for {league_code}")
        except Exception as e:
            if self.debug: print(f"⚠️ league {league_code} calibrators not found: {e}")
            iso = None
        self._iso_league_cache[league_code] = iso
        return iso

    def _apply_iso(self, val: float, iso_obj) -> float:
        try:
            return float(iso_obj.predict([val])[0])
        except Exception:
            # fallback: clip
            return max(0.001, min(0.999, val))
    
    def recommend(self, pred: dict) -> list:
        """
        Restituisce una lista di suggerimenti di mercato basati sulle soglie ottimali
        """
        recs = []
        
        if pred["p_over_0_5"] >= DEFAULT_THRESHOLDS["O0_5"]:
            recs.append({
                "market": "Over 0.5",
                "confidence": pred["p_over_0_5"],
                "threshold": DEFAULT_THRESHOLDS["O0_5"]
            })

        if pred["p_over_1_5"] >= DEFAULT_THRESHOLDS["O1_5"]:
            recs.append({
                "market": "Over 1.5",
                "confidence": pred["p_over_1_5"],
                "threshold": DEFAULT_THRESHOLDS["O1_5"]
            })

        if pred["p_over_2_5"] >= DEFAULT_THRESHOLDS["O2_5"]:
            recs.append({
                "market": "Over 2.5",
                "confidence": pred["p_over_2_5"],
                "threshold": DEFAULT_THRESHOLDS["O2_5"],
                "ev": pred.get("ev_over_2_5"),
                "ev_flag": pred.get("ev_over_2_5", 0) > 0.02
            })

        return recs

    def predict(self, match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        match atteso: {
          "Avg>2.5": float, "Avg<2.5": float,
          "league_code": str (opzionale)
        }
        """
        O = match.get("Avg>2.5")
        U = match.get("Avg<2.5")
        p_over25_fair = self._fair_over_prob(O, U)
        if p_over25_fair is None:
            return None

        lam = _solve_lambda_from_pover25(p_over25_fair)
        probs = _poisson_over_probs(lam)  # grezze

        # scegli calibratori (league → global fallback)
        league_code = match.get("league_code")
        iso = None
        if league_code:
            iso = self._load_league_calibs(league_code)

        if iso is not None:
            iso05, iso15, iso25 = iso.get("05"), iso.get("15"), iso.get("25")
        else:
            iso05, iso15, iso25 = self._iso_g.get("05"), self._iso_g.get("15"), self._iso_g.get("25")

        p05c = self._apply_iso(probs["p05"], iso05) if iso05 else probs["p05"]
        p15c = self._apply_iso(probs["p15"], iso15) if iso15 else probs["p15"]
        p25c = self._apply_iso(probs["p25"], iso25) if iso25 else probs["p25"]

        # EV solo per O2.5 (non avendo le altre quote)
        ev25 = p25c * float(O) - 1.0 if O and O > 0 else None

        out = {
            "lambda_market": lam,
            "p_over_0_5_raw": probs["p05"],
            "p_over_1_5_raw": probs["p15"],
            "p_over_2_5_raw": probs["p25"],
            "p_over_0_5": p05c,
            "p_over_1_5": p15c,
            "p_over_2_5": p25c,
            "ev_over_2_5": ev25,
            "suggest_over_2_5": (ev25 is not None and ev25 > 0.02),
            "model": "OverModelV1",
        }
        if self.debug:
            print(f"[OverModelV1] λ={lam:.3f} | p25_fair={p_over25_fair:.3f} | "
                  f"p25_raw={probs['p25']:.3f} → p25_cal={p25c:.3f} | EV={ev25:.3f if ev25 is not None else None}")
        return out

