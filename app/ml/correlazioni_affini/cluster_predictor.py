# app/ml/correlazioni_affini/cluster_predictor.py

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

from .cluster_engine import CLUSTER_ENGINE


def predict_cluster_and_profile(row_dict: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Wrapper comodo per l'API:
      - row_dict: dizionario con alcune feature (es. bk_p1, bk_px, bk_p2, bk_pO25, bk_pU25, ecc.)
      - riempie i buchi con le medie di colonna
      - restituisce (cluster_id, profilo_cluster)
    """
    return CLUSTER_ENGINE.predict_cluster_with_profile(row_dict)


def predict_cluster_only(row_dict: Dict[str, Any]) -> int:
    """
    Solo cluster_id, senza profilo (se mai ti serve).
    """
    return CLUSTER_ENGINE.predict_cluster(row_dict)