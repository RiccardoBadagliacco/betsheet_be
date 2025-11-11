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
        return {}

    # 1️⃣ Ordina quote reali e bookmaker per determinare la scala
    scala_reale = "".join([k for k, _ in sorted(quota_reale_allibrata.items(), key=lambda x: x[1])])
    print("Scala reale:", scala_reale)
    scala_book = "".join([k for k, _ in sorted(odds.items(), key=lambda x: x[1])])
    print("Scala bookmaker:", scala_book)

    favorito = scala_reale[0]
    q_fav = quota_reale_allibrata.get(favorito, 0)
    spalm_fav = spalmatura.get(favorito, 0)

    scala_coerente = scala_reale == scala_book

    # 3️⃣ Se non rispetta le condizioni, non restituisce nulla
    if not (scala_coerente):
        return {}

    # 4️⃣ Output sintetico
    return {
        "label": f"Scala {scala_reale}"
    }