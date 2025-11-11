# metrics.py
from typing import Dict, Any
from .metodo_favorita import get_metodo_favorita


def get_metrics(picchetti: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggrega tutte le metriche disponibili (solo se applicabili).
    """
    metrics = {}
    if "1X2" in picchetti:
        favorita = get_metodo_favorita(picchetti["1X2"])
        print("Metodo Favorita:", favorita)
        if favorita:
            metrics["favorita"] = favorita

    # In futuro: aggiungi altri metodi qui (over/under, value, ecc.)
    return metrics