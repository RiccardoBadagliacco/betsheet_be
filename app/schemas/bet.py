from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from datetime import datetime


class BackrollBase(BaseModel):
    id: str
    backroll: Optional[float] = None
    cassa: Optional[float] = None
    name: Optional[str] = None


class BackrollCreate(BaseModel):
    id: Optional[str] = None
    backroll: Optional[float] = None
    cassa: Optional[float] = None
    name: Optional[str] = None


class BackrollUpdate(BaseModel):
    backroll: Optional[float] = None
    cassa: Optional[float] = None
    name: Optional[str] = None


class BackrollRead(BackrollBase):
    bets: Optional[List["BetRead"]] = None
    # computed field: initial backroll + sum of bet profits
    current_backroll: Optional[float] = None
    # current cassa taken from the last bet (or fallback to backroll.cassa)
    current_cassa: Optional[float] = None

    # overall profit and percentage
    profitto: Optional[float] = None
    profitto_percent: Optional[float] = None
    # win rate percentage (Vinto / Giocate * 100)
    win_rate: Optional[float] = None

    # stats object (labels in Italian)
    stats: Optional[Dict[str, int]] = None

    # history: list of {date: YYYY-MM-DD, backroll: float}
    class HistoryItem(BaseModel):
        date: str
        backroll: float

    history: Optional[List[HistoryItem]] = None

    class Config:
        orm_mode = True


class BetBase(BaseModel):
    id: str
    backroll_id: Optional[str] = None
    data: Optional[str] = None
    backroll_iniziale: Optional[float] = None
    cassa: Optional[float] = None
    quota: Optional[float] = None
    stake: Optional[float] = None
    importo: Optional[float] = None
    vincita: Optional[float] = None
    backroll_finale: Optional[float] = None
    profitto_totale: Optional[float] = None
    profitto: Optional[float] = None
    esito: Optional[str] = None
    partite: Optional[Any] = None
    tipo_giocata: Optional[str] = None
    link: Optional[str] = None
    note: Optional[str] = None
    created_at: Optional[datetime] = None
    backroll_name: Optional[str] = None
    # percentage profit for this bet relative to importo
    profitto_percent: Optional[float] = None


class BetCreate(BaseModel):
    # when creating, id is optional (generated if not provided)
    id: Optional[str] = None
    backroll_id: Optional[str] = None
    data: Optional[str] = None
    backroll_iniziale: Optional[float] = None
    cassa: Optional[float] = None
    quota: Optional[float] = None
    stake: Optional[float] = None
    importo: Optional[float] = None
    vincita: Optional[float] = None
    backroll_finale: Optional[float] = None
    profitto_totale: Optional[float] = None
    profitto: Optional[float] = None
    esito: Optional[str] = None
    partite: Optional[Any] = None
    tipo_giocata: Optional[str] = None
    link: Optional[str] = None
    note: Optional[str] = None


class BetRead(BetBase):
    class Config:
        orm_mode = True


class BetUpdate(BaseModel):
    backroll_id: Optional[str] = None
    data: Optional[str] = None
    backroll_iniziale: Optional[float] = None
    cassa: Optional[float] = None
    quota: Optional[float] = None
    stake: Optional[float] = None
    importo: Optional[float] = None
    vincita: Optional[float] = None
    backroll_finale: Optional[float] = None
    profitto_totale: Optional[float] = None
    profitto: Optional[float] = None
    esito: Optional[str] = None
    partite: Optional[Any] = None
    tipo_giocata: Optional[str] = None
    link: Optional[str] = None
    note: Optional[str] = None


# Resolve forward refs
BackrollRead.update_forward_refs()
