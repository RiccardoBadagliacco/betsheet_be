# metodo_favorita.py
from typing import Dict, Any

def get_metodo_favorita(picchetto: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analizza il 'Metodo della Favorita' (Stats4Bets-style).
    Restituisce una label sintetica (es. "Scala 2X1") solo se il metodo è applicabile.
    """
    quota_reale_allibrata = picchetto.get("quota_reale_allibrata", {})
    odds = picchetto.get("quota_bookmaker", {})
    spalmatura = picchetto.get("spalmatura_allibramento%", {})

    if not quota_reale_allibrata or not odds:
        print("Dati insufficienti per Metodo della Favorita.")
        return {}

    # 🧩 Safe cast per evitare TypeError durante il sort
    def safe_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    quota_reale_allibrata = {k: safe_float(v) for k, v in quota_reale_allibrata.items() if safe_float(v) is not None}
    odds = {k: safe_float(v) for k, v in odds.items() if safe_float(v) is not None}

    if len(quota_reale_allibrata) < 3 or len(odds) < 3:
        print("Quote incomplete o non numeriche, skip.")
        return {}

    # 1️⃣ Ordina quote reali e bookmaker per determinare la scala
    scala_reale = "".join([k for k, _ in sorted(quota_reale_allibrata.items(), key=lambda x: x[1])])
    scala_book = "".join([k for k, _ in sorted(odds.items(), key=lambda x: x[1])])
    scala_coerente = scala_reale == scala_book

    # 3️⃣ Se non rispetta le condizioni, non restituisce nulla
    if not scala_coerente:
        print("Scala non coerente, nessun output.")
        return {}

    # 4️⃣ Output sintetico
    print("Metodo della Favorita applicabile:", f"Scala {scala_reale}")
    return {
        "label": f"Scala {scala_reale}"
    }