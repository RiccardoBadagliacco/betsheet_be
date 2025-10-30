"""
==============================================================
Betting Config — Refactor 2025
==============================================================

Obiettivi:
- Centralizzare soglie di confidenza e parametri per mercati
- Rimuovere duplicazioni con betting_recommendations
- Facilitare tuning automatico da backtest
- Mantenere compatibilità con context_scoring_v4 e ml_football_exact
"""

# --------------------------------------------------------------
# Mercati target principali (usati per raccomandazioni)
# --------------------------------------------------------------
TARGET_MARKETS = [
    "Over 0.5 Goal",
    "Over 1.5 Goal",
    "Over 2.5 Goal",
    "Under 2.5 Goal",
    "Multigol Casa 1-3",
    "Multigol Casa 1-4",
    "Multigol Casa 1-5",
    "Multigol Ospite 1-3",
    "Multigol Ospite 1-4",
    "Multigol Ospite 1-5",
]

# --------------------------------------------------------------
# Mercati candidati (per filtraggio e threshold tuning)
# --------------------------------------------------------------
CANDIDATE_MARKETS = list(TARGET_MARKETS) + [
    "Under 0.5 Goal",
    "Over 3.5 Goal",
    "1X2 Casa",
    "1X2 Ospite",
    "Doppia Chance 1X",
    "Doppia Chance X2",
    "Doppia Chance 12",
]

# --------------------------------------------------------------
# Soglie minime di confidenza per raccomandazioni betting
# (valori ottimizzabili via backtest / tuning dinamico)
# --------------------------------------------------------------
MARKET_THRESHOLDS = {
    "Multigol Casa 1-3": 70,
  "Multigol Casa 1-4": 80,
  "Multigol Casa 1-5": 80,
  "Multigol Ospite 1-3": 70,
  "Multigol Ospite 1-4": 75,
  "Multigol Ospite 1-5": 95,
  "Over 0.5 Goal": 90,
  "Over 1.5 Goal": 75
}

# --------------------------------------------------------------
# Mappatura chiavi tecniche (model) → mercati leggibili (config)
# --------------------------------------------------------------
MARKET_KEY_MAP = {
    "O_0_5": "Over 0.5 Goal",
    "O_1_5": "Over 1.5 Goal",
    "O_2_5": "Over 2.5 Goal",
    "U_2_5": "Under 2.5 Goal",
    "MG_Casa_1_3": "Multigol Casa 1-3",
    "MG_Casa_1_4": "Multigol Casa 1-4",
    "MG_Casa_1_5": "Multigol Casa 1-5",
    "MG_Ospite_1_3": "Multigol Ospite 1-3",
    "MG_Ospite_1_4": "Multigol Ospite 1-4",
    "MG_Ospite_1_5": "Multigol Ospite 1-5",
    "1X2_H": "1X2 Casa",
    "1X2_A": "1X2 Ospite",
    "1X2_D": "1X2 X",
}

# --------------------------------------------------------------
# Altri parametri globali
# --------------------------------------------------------------

# Confidenza minima globale per proporre una giocata (overrideabile per mercato)
MIN_CONFIDENCE = 0.55

# Tag per clustering di mercati (utile per combinazioni e filtri contestuali)
MARKET_GROUPS = {
    "over": ["Over 0.5 Goal", "Over 1.5 Goal", "Over 2.5 Goal", "Over 3.5 Goal"],
    "under": ["Under 0.5 Goal", "Under 2.5 Goal"],
    "mg_home": ["Multigol Casa 1-3", "Multigol Casa 1-4", "Multigol Casa 1-5"],
    "mg_away": ["Multigol Ospite 1-3", "Multigol Ospite 1-4", "Multigol Ospite 1-5"],
    "1x2": ["1X2 Casa", "1X2 Ospite"],
    "doppia_chance": ["Doppia Chance 1X", "Doppia Chance X2", "Doppia Chance 12"],
}

# Confidenza alta (usata per ranking visivo interfacce)
HIGH_CONFIDENCE_THRESHOLD = 0.70

# Confidenza media (per segmentazione suggerimenti)
MEDIUM_CONFIDENCE_THRESHOLD = 0.60

# Mercati da ignorare se la quota è assente o anomala
IGNORE_IF_NO_ODDS = [
    "1X2 Casa",
    "1X2 Ospite",
    "Doppia Chance 1X",
    "Doppia Chance X2",
    "Doppia Chance 12",
]

# Mercati per i quali preferiamo quote reali rispetto a probabilità modello
PREFER_REAL_ODDS = [
    "1X2 Casa",
    "1X2 Ospite",
    "Doppia Chance 1X",
    "Doppia Chance X2",
    "Doppia Chance 12",
]

# Mercati supportati per boost contestuali (context_scoring_v4)
CONTEXT_ENABLED_MARKETS = list(TARGET_MARKETS)