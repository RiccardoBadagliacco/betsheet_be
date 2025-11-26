# -*- coding: utf-8 -*-
"""
BET TRANSLATION ENGINE (v2.1 con statistiche integrate)
-------------------------------------------------------
Compatibile con:
- Regole 1X2
- Regole MULTIGOAL
- Regole UNDER/OVER
- Backtest correlazioni_affini_v2
"""

import json
import os
from pathlib import Path

# ==========================================================
# CARICA STATISTICHE GLOBALI DEI BET (generate dal backtest)
# ==========================================================

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_BET_STATS_FILE = _DATA_DIR / "step5_bet_global_stats.json"

try:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    stats_path = os.path.join(_BET_STATS_FILE)

    print("📌 Loading bet_stats.json from:", stats_path)

    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            data = json.load(f)

        BET_GLOBAL_STATS = data.get("global", {})
        print("📌 BET_GLOBAL_STATS loaded. Keys:", BET_GLOBAL_STATS.keys())
    else:
        print("❌ bet_stats.json NOT FOUND at:", stats_path)
except Exception as e:
    print("⚠️ Error loading bet_stats.json:", e)




# ======================================================================
# TRADUZIONE BET-TAG → DESCRIZIONE CHIARA
# ======================================================================

def translate_single_bet(tag: str, side: str) -> str:
    is_home = (side == "home")

    # -------------------------------
    # DOPPIA CHANCE
    # -------------------------------
    if tag == "BET_DC_FAV":
        return "Doppia Chance 1X" if is_home else "Doppia Chance X2"

    if tag == "BET_DC_OPPOSITE":
        return "Doppia Chance X2" if is_home else "Doppia Chance 1X"

    # -------------------------------
    # GOAL FAVORITA
    # -------------------------------
    if tag == "BET_FAV_OVER_0_5":
        return "Favorita segna almeno 1 gol"

    if tag == "BET_FAV_SEGNA":
        return "Favorita segna (Yes)"

    # -------------------------------
    # OVER/UNDER
    # -------------------------------
    if tag == "BET_OVER15":
        return "Over 1.5 totale"

    if tag == "BET_OVER25":
        return "Over 2.5 totale"

    if tag == "BET_UNDER15":
        return "Under 1.5 totale"

    if tag == "BET_UNDER25":
        return "Under 2.5 totale"

    # -------------------------------
    # MULTIGOAL FAVORITA
    # -------------------------------
    if tag == "BET_FAV_1_3":
        return "Multigol Favorita 1–3"

    if tag == "BET_FAV_1_4":
        return "Multigol Favorita 1–4"

    if tag == "BET_FAV_1_5":
        return "Multigol Favorita 1–5"

    # -------------------------------
    # UNDER SPECIALI
    # -------------------------------
    if tag == "BET_UNDER_1_5_FAV":
        return "Favorita meno di 2 gol"

    if tag == "BET_LAY_FAV_SEGNA":
        return "Favorita NON segna"

    # -------------------------------
    # SEGNI SECCHI
    # -------------------------------
    if tag == "BET_1":
        return "Segno 1"

    if tag == "BET_2":
        return "Segno 2"

    # -------------------------------
    # NO-BET / WARNING
    # -------------------------------
    if tag == "BET_NO_BET_STRONG":
        return "Evita giocata (contesto rischioso)"

    if tag == "BET_NO_BET":
        return "Nessuna giocata consigliata"

    print(f"Warning: Unrecognized bet tag '{tag}'")
    return tag.replace("_", " ")


# ======================================================================
# MOTIVAZIONI BREVI PER OGNI TIPO DI BET
# ======================================================================

BET_REASON = {
    "Doppia Chance 1X": "Protegge contro il pareggio e stabilizza il rischio.",
    "Doppia Chance X2": "Copre dalla sconfitta della favorita in contesti instabili.",
    "Favorita segna almeno 1 gol": "Trend offensivo positivo nelle partite affini.",
    "Favorita segna (Yes)": "La favorita trova spesso almeno una rete in contesti simili.",
    "Over 1.5 totale": "Alta probabilità di almeno due gol complessivi.",
    "Over 2.5 totale": "Match molto offensivo nei pattern storici analizzati.",
    "Under 1.5 totale": "Partite chiuse, ritmo basso e poche occasioni.",
    "Under 2.5 totale": "Storicamente match con meno di tre reti.",
    "Multigol Favorita 1–3": "Range realizzativo più frequente della favorita.",
    "Multigol Favorita 1–4": "La favorita produce molte occasioni nei precedenti.",
    "Multigol Favorita 1–5": "Profilo molto offensivo e continuo nel tempo.",
    "Favorita meno di 2 gol": "La favorita produce poco in contesti simili.",
    "Favorita NON segna": "Alta frequenza di No-Goal favorita nei dati affini.",
    "Segno 1": "La favorita di casa mostra forte superiorità.",
    "Segno 2": "La favorita ospite domina statisticamente in scenari analoghi.",
    "Evita giocata (contesto rischioso)": "Troppi indicatori contrari privi di coerenza.",
    "Nessuna giocata consigliata": "Indicatori contrastanti, match non giocabile.",
}


# ======================================================================
# PRIORITÀ PER ORDINARE I BET
# ======================================================================

BET_PRIORITY = {
    "Doppia Chance 1X": 1,
    "Doppia Chance X2": 1,
    "Favorita segna almeno 1 gol": 2,
    "Favorita segna (Yes)": 2,
    "Over 1.5 totale": 2,
    "Multigol Favorita 1–3": 3,
    "Multigol Favorita 1–4": 3,
    "Multigol Favorita 1–5": 3,
    "Segno 1": 4,
    "Segno 2": 4,
    "Over 2.5 totale": 5,
    "Under 2.5 totale": 5,
    "Under 1.5 totale": 5,
    "Favorita meno di 2 gol": 5,
    "Favorita NON segna": 5,
    "Nessuna giocata consigliata": 10,
    "Evita giocata (contesto rischioso)": 10,
}


# ======================================================================
# CONFIDENCE
# ======================================================================

def map_severity_to_confidence(severity: str) -> str:
    if severity == "high":
        return "Alta confidenza"
    if severity == "medium":
        return "Media confidenza"
    return "Bassa confidenza"


# ======================================================================
# BUILD SUGGESTIONS — AGGIUNTA STATISTICHE QUI 🔥
# ======================================================================

def build_bet_suggestions(bet_tags, severity: str, side: str):
    suggestions = []

    for tag in bet_tags:
        label = translate_single_bet(tag, side)
        reason = BET_REASON.get(label, "Coerente con i pattern storici.")
        priority = BET_PRIORITY.get(label, 999)

        # 🔥 AGGIUNGIAMO LE STATISTICHE GLOBALI SE DISPONIBILI
        stats = BET_GLOBAL_STATS.get(tag)

        suggestions.append({
            "tag_raw": tag,
            "label": label,
            "reason": reason,
            "priority": priority,
            "stats": stats,   # ← aggiunta fondamentale
        })

    # Ordine per priorità
    suggestions = sorted(suggestions, key=lambda x: x["priority"])

    # Limito a max 4 suggerimenti
    suggestions = suggestions[:4]

    return {
        "confidence": map_severity_to_confidence(severity),
        "suggestions": suggestions
    }
