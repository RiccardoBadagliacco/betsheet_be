# metrics.py
from typing import Dict, Any
from .metodo_favorita import get_metodo_favorita


def get_metrics(picchetti: Dict[str, Any]) -> Dict[str, Any]:
    """Aggrega tutte le metriche disponibili (solo se applicabili)."""
    metrics: Dict[str, Any] = {}

    picchetto_1x2 = picchetti.get("1X2")
    if picchetto_1x2:
        favorita = get_metodo_favorita(picchetto_1x2)
        if favorita:
            metrics["1X2"] = favorita

    # In futuro: aggiungi altri metodi qui (over/under, value, ecc.)
    return metrics
