from typing import List, Dict, Any, Optional
import math
import numbers

def _sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, numbers.Real):
        return value
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    return value


class BettingAlert:
    """
    Pagina FE-ready dell’alert:
    - bets = oggetti suggeriti già tradotti
    - giocate_suggerite = stessa lista
    - tags = tag tecnici (per debug/backtest)
    - meta = contiene anche bet_tags_raw (lista raw dei BET_*)
    """
    def __init__(
        self,
        code: str,
        severity: str,
        message: str,
        tags: Optional[List[str]] = None,
        bets: Optional[List[Any]] = None,  # ORA: lista di oggetti, non stringhe
        meta: Optional[Dict[str, Any]] = None,
        tag_descriptions: Optional[Dict[str, str]] = None,
    ):
        self.code = code
        self.severity = severity
        self.message = message
        self.tags = tags or []
        self.bets = bets or []              # QUI: lista di oggetti bet tradotti
        self.meta = meta or {}
        self.tag_descriptions = tag_descriptions or {}

    def to_output(self):
        """
        Il FE ora riceve direttamente:
        - bets = lista oggetti tradotti
        - giocate_suggerite = stessa lista
        """
        return _sanitize_json({
            "alert": self.code,
            "severity": self.severity,
            "description": self.message,
            "tags": self.tags or [],
            "tag_descriptions": self.tag_descriptions or {},
            "giocate_suggerite": self.bets,  # già tradotte
            "bets": self.bets,               # FE-friendly
            "meta": self.meta or {}
        })