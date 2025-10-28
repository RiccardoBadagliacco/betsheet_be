"""
================================================================
Betting Recommendations — Refactor + custom thresholds (merged)
================================================================
- Legge soglie statiche da betting_config.py
- (Ri)legge soglie dinamiche da optimal_thresholds.json se presente
- get_threshold con soft offset per Over/Multigol (come versione copy)
- Supporta ctx_skip / ctx_force
- Include logica Multigol baseline
- (Opzionale) include 1X2 / DC nelle raccomandazioni
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from addons.betting_config import (
    MARKET_THRESHOLDS,           # decimali (0.70, 0.65, ...)
    MIN_CONFIDENCE,             # decimale
    TARGET_MARKETS,
    CANDIDATE_MARKETS,          # utile per future estensioni
    HIGH_CONFIDENCE_THRESHOLD,  # decimale
    MEDIUM_CONFIDENCE_THRESHOLD,# decimale
)

# ==============================
# Opzioni
# ==============================
INCLUDE_1X2_AND_DC = True  # riabilita 1X2 / DC come nella versione copy
DEBUG_BETS = False         # metti True per stampare passaggi di debug

# ==============================
# Caricamento soglie dinamiche
# ==============================
THRESHOLD_FILE = Path("optimal_thresholds.json")
USE_CUSTOM_THRESHOLDS = True
_optimal_thresholds: Dict[str, Any] = {}

if THRESHOLD_FILE.exists():
    try:
        with open(THRESHOLD_FILE) as f:
            raw = json.load(f)
        # compatibile sia con {"Over 1.5 Goal": 75} sia con {"Over 1.5 Goal": {"threshold": 75, ...}}
        for k, v in raw.items():
            if isinstance(v, dict) and "threshold" in v:
                _optimal_thresholds[k] = int(v["threshold"])
            else:
                _optimal_thresholds[k] = int(v)
        if DEBUG_BETS:
            print(f"[THR LOADER] Caricate soglie custom per {len(_optimal_thresholds)} mercati")
    except Exception as e:
        print(f"[THR LOADER] ⚠️ Errore caricando soglie personalizzate: {e}")
else:
    if DEBUG_BETS:
        print(f"[THR LOADER] File {THRESHOLD_FILE} non trovato — uso sole soglie di config")

def get_threshold(label: str) -> int:
    """
    Restituisce la soglia (0-100) per un mercato.
    Priorità: custom JSON -> config -> default globale.
    Applica offset soft -5 per Over/Multigol (comportamento legacy utile).
    """
    # 1) custom file
    if USE_CUSTOM_THRESHOLDS and label in _optimal_thresholds:
        thr = int(_optimal_thresholds[label])
        if label.startswith("Over") or "Multigol" in label:
            thr -= 5   # offset soft legacy
        return thr

    # 2) config decimale
    if label in MARKET_THRESHOLDS:
        return int(round(MARKET_THRESHOLDS[label] * 100))

    # 3) fallback ragionevole
    return int(round(MIN_CONFIDENCE * 100))

# ==============================
# Helpers
# ==============================

def _calc_confidence(prediction: Dict[str, Any], label: str) -> float:
    """
    Legge la probabilità (0..1) per il mercato 'label' già in forma umana
    (es. 'Over 1.5 Goal', 'Multigol Casa 1-3', ...) e restituisce 0..100.
    """
    p = prediction.get(label)
    if p is None:
        # compat: alcuni setup potrebbero avere chiavi alternative residue
        if label.startswith("Over "):
            # es.: "Over 1.5 Goal" -> chiave alternativa "O_1_5"
            try:
                parts = label.split(" ")[1]  # "1.5"
                code = f"O_{parts.replace('.','_')}"
                p = prediction.get(code)
            except Exception:
                p = None
        elif label.startswith("Under "):
            try:
                parts = label.split(" ")[1]  # "2.5"
                code = f"U_{parts.replace('.','_')}"
                p = prediction.get(code)
            except Exception:
                p = None

    try:
        return round(float(p) * 100, 2) if p is not None else 0.0
    except Exception:
        return 0.0

def _apply_ctx_delta(confidence: float, label: str, ctx: Dict[str, Any]) -> float:
    if not ctx:
        return confidence
    delta_map = ctx.get("ctx_delta_thresholds", {}) or {}
    if label in delta_map:
        confidence += float(delta_map[label])
    # clamp
    return max(0.0, min(confidence, 100.0))

# ==============================
# Multigol baseline (legacy utile)
# ==============================

def _multigol_baseline(pred: Dict[str, Any], quotes: Optional[Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Logica storica: decide se spingere Casa/Ospite in base a favorito.
    Usa quote reali se presenti, altrimenti somma prob. 1X2.
    """
    recs: List[Dict[str, Any]] = []

    # Probabilità 1X2 (0..100)
    pH = float(pred.get("1X2_H", pred.get("prob_home", 0))) * 100
    pD = float(pred.get("1X2_D", pred.get("prob_draw", 0))) * 100
    pA = float(pred.get("1X2_A", pred.get("prob_away", 0))) * 100

    # Lambdas
    lH = float(pred.get("lambda_home", pred.get("home_goals", 0)))
    lA = float(pred.get("lambda_away", pred.get("away_goals", 0)))

    favorite_team: Optional[str] = None
    if quotes:
        qH = quotes.get("1", 999)
        qA = quotes.get("2", 999)
        if abs(qH - qA) >= 0.2:
            if qH < qA and qH <= 1.70:
                favorite_team = "home"
            elif qA < qH and qA <= 1.90:
                favorite_team = "away"
    else:
        if abs(pH - pA) >= 5.0:
            favorite_team = "home" if pH > pA else "away"

    def maybe_add(lbl: str, market: str, prob_01: float):
        c = float(prob_01) * 100.0
        thr = get_threshold(lbl)
        if c >= thr:
            recs.append({
                "market": market,
                "prediction": market,
                "confidence": round(c, 1),
                "threshold": thr,
            })

    if favorite_team == "home" and lH >= 1.0:
        maybe_add("Multigol Casa 1-3", "Multigol Casa 1-3", float(pred.get("Multigol Casa 1-3", pred.get("MG_Casa_1_3", 0))))
        maybe_add("Multigol Casa 1-4", "Multigol Casa 1-4", float(pred.get("Multigol Casa 1-4", pred.get("MG_Casa_1_4", 0))))
        maybe_add("Multigol Casa 1-5", "Multigol Casa 1-5", float(pred.get("Multigol Casa 1-5", pred.get("MG_Casa_1_5", 0))))
    if favorite_team == "away" and lA >= 1.0:
        maybe_add("Multigol Ospite 1-3", "Multigol Ospite 1-3", float(pred.get("Multigol Ospite 1-3", pred.get("MG_Ospite_1_3", 0))))
        maybe_add("Multigol Ospite 1-4", "Multigol Ospite 1-4", float(pred.get("Multigol Ospite 1-4", pred.get("MG_Ospite_1_4", 0))))
        maybe_add("Multigol Ospite 1-5", "Multigol Ospite 1-5", float(pred.get("Multigol Ospite 1-5", pred.get("MG_Ospite_1_5", 0))))

    return recs

def adjust_threshold_for_context(label: str, thr: int, prediction: Dict[str, Any]) -> int:
    """
    Modifica dinamicamente la soglia in base al contesto pre-match.
    Non sostituisce la soglia statica ma la aggiusta.
    """
    # estrai info chiave dal modello
    lambda_home = prediction.get("lambda_home", 0)
    lambda_away = prediction.get("lambda_away", 0)
    total_lambda = lambda_home + lambda_away

    pH = float(prediction.get("1X2_H", prediction.get("prob_home", 0)))
    pA = float(prediction.get("1X2_A", prediction.get("prob_away", 0)))
    strength_diff = abs(pH - pA) * 100  # differenza percentuale “forza”

    # ⚽ esempio base: over/multigol più facili se partita è offensiva
    if label.startswith("Over") or "Multigol" in label:
        if total_lambda >= 3.0:
            thr -= 10  # partita molto offensiva → abbassa soglia
        elif total_lambda <= 2.0:
            thr += 5   # partita con pochi goal attesi → alza soglia

    # 🧭 big match / equilibrio → soglia più severa
    if strength_diff < 5:  # match molto equilibrato
        thr += 5

    # Squadra molto favorita → soglia più permissiva
    if strength_diff >= 20:
        thr -= 5

    # clamp finale
    thr = max(40, min(thr, 95))
    return thr

# ==============================
# API principale
# ==============================

def get_recommended_bets(prediction: Dict[str, Any], quotes: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []
    print('------> get_recommended_bets called')
    # Context directives (compat legacy)
    ctx_delta = prediction.get("ctx_delta_thresholds", {}) or {}
    ctx_skip  = set(prediction.get("skip_recommendations", []) or [])
    ctx_force = set(prediction.get("force_recommendations", []) or [])

    # 1) Mercati “standard” (TARGET_MARKETS)
    already = set()
    for label in TARGET_MARKETS:
        if label in ctx_skip:
            continue
        conf = _calc_confidence(prediction, label)  # 0..100
        conf = _apply_ctx_delta(conf, label, {"ctx_delta_thresholds": ctx_delta})
        thr = get_threshold(label)
        thr = adjust_threshold_for_context(label, thr, prediction)

        forced = (label in ctx_force)
        if forced or conf >= thr:
            recommendations.append({
                "market": label,
                "prediction": label,
                "confidence": round(conf, 1),
                "threshold": thr,
                "forced": forced,
                "high_confidence": conf >= HIGH_CONFIDENCE_THRESHOLD * 100,
                "medium_confidence": conf >= MEDIUM_CONFIDENCE_THRESHOLD * 100,
            })
            already.add(label)

    # 2) Multigol baseline (aggiunge solo dove non già presente)
    for rec in _multigol_baseline(prediction, quotes):
        if rec["market"] in ctx_skip or rec["market"] in already:
            continue
        # delta/force
        lbl = rec["market"]
        conf = rec["confidence"]
        thr = rec["threshold"]
        if lbl in ctx_force:
            rec["forced"] = True
        # applica eventuale delta
        conf = _apply_ctx_delta(conf, lbl, {"ctx_delta_thresholds": ctx_delta})
        rec["confidence"] = round(conf, 1)
        rec["high_confidence"] = conf >= HIGH_CONFIDENCE_THRESHOLD * 100
        rec["medium_confidence"] = conf >= MEDIUM_CONFIDENCE_THRESHOLD * 100
        recommendations.append(rec)
        already.add(lbl)

    # 3) (Opzionale) 1X2 / DC – come in versione copy
    if INCLUDE_1X2_AND_DC:
        prob_home = float(prediction.get('1X2_H', prediction.get('prob_home', 0))) * 100
        prob_draw = float(prediction.get('1X2_D', prediction.get('prob_draw', 0))) * 100
        prob_away = float(prediction.get('1X2_A', prediction.get('prob_away', 0))) * 100

        def maybe_add_1x2(lbl: str, c: float):
            if lbl in ctx_skip:
                return
            thr = get_threshold(lbl)
            c = _apply_ctx_delta(c, lbl, {"ctx_delta_thresholds": ctx_delta})
            forced = lbl in ctx_force
            if forced or c >= thr:
                recommendations.append({
                    "market": lbl,
                    "prediction": lbl,
                    "confidence": round(c, 1),
                    "threshold": thr,
                    "forced": forced,
                    "high_confidence": c >= HIGH_CONFIDENCE_THRESHOLD * 100,
                    "medium_confidence": c >= MEDIUM_CONFIDENCE_THRESHOLD * 100,
                })

        if prob_home: maybe_add_1x2("1X2 Casa", prob_home)
        if prob_draw: maybe_add_1x2("1X2 X", prob_draw)
        if prob_away: maybe_add_1x2("1X2 Ospite", prob_away)

        # DC
        def sumc(a, b): return a + b if a and b else 0.0
        prob_1x = sumc(prob_home, prob_draw)
        prob_x2 = sumc(prob_draw, prob_away)
        prob_12 = sumc(prob_home, prob_away)
        for lbl, c in [("Doppia Chance 1X", prob_1x), ("Doppia Chance X2", prob_x2), ("Doppia Chance 12", prob_12)]:
            if c == 0 or lbl in ctx_skip:
                continue
            thr = get_threshold(lbl)
            c = _apply_ctx_delta(c, lbl, {"ctx_delta_thresholds": ctx_delta})
            forced = lbl in ctx_force
            if forced or c >= thr:
                recommendations.append({
                    "market": lbl,
                    "prediction": lbl,
                    "confidence": round(c, 1),
                    "threshold": thr,
                    "forced": forced,
                    "high_confidence": c >= HIGH_CONFIDENCE_THRESHOLD * 100,
                    "medium_confidence": c >= MEDIUM_CONFIDENCE_THRESHOLD * 100,
                })

    # Ordina per confidenza discendente
    recommendations.sort(key=lambda x: x["confidence"], reverse=True)
    return recommendations
