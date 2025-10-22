from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Bet, Backroll
from app.schemas.bet import BetRead
from app.schemas.bet import BetCreate
import uuid
import json
from datetime import datetime, timedelta

router = APIRouter()


def _safe_parse_json(val: str):
    try:
        return json.loads(val)
    except Exception:
        return val


def _normalize_esito(esito: Optional[str]) -> Optional[str]:
    if esito is None:
        return None
    return esito.strip().lower()


def _canonicalize_esito(esito: Optional[str]) -> Optional[str]:
    """Map various esito representations to canonical 'vinta'/'persa'/'in_progress'."""
    if esito is None:
        return None
    s = str(esito).strip().lower()
    if s in ("v", "1", "vinto", "vinta", "win", "won", "vinta"):
        return "vinta"
    if s in ("l", "0", "perso", "persa", "loss", "lost", "persa"):
        return "persa"
    if any(tok in s for tok in ("in progress", "in_progress", "pending", "p", "inprogress")):
        return "in_progress"
    # allow already canonical
    if s in ("vinta", "persa", "in_progress"):
        return s
    return None


def _detect_single_match_outcome(match) -> Optional[str]:
    """Return 'won', 'lost', 'in_progress' or None for a single match object or value."""
    if match is None:
        return None
    # if it's a dict, try common keys
    val = None
    if isinstance(match, dict):
        for k in ("esito", "esito_match", "result", "outcome", "status", "stato", "risultato"):
            if k in match and match[k] is not None:
                val = match[k]
                break
    else:
        val = match

    if val is None:
        return None

    s = str(val).strip().lower()
    # numeric or textual codes -> map to canonical tokens
    if s in ("1", "v", "vinto", "vinta", "win", "won"):
        return "vinta"
    if s in ("0", "l", "perso", "persa", "loss", "lost"):
        return "persa"
    if any(tok in s for tok in ("in progress", "in_progress", "pending", "p")):
        return "in_progress"
    # try to catch italian words
    if "vinto" in s or "vinta" in s:
        return "vinta"
    if "perso" in s or "persa" in s:
        return "persa"

    return None


def _map_count_to_tipo_giocata(n: int) -> str:
    if n <= 1:
        return "singola"
    if n == 2:
        return "doppia"
    if n == 3:
        return "tris"
    if n == 4:
        return "poker"
    if 5 <= n <= 8:
        return "multipla"
    return "listone"


def _normalize_partite_and_infer(partite_val):
    """Normalize `partite` input into a list and infer tipo_giocata and esito.

    Returns tuple: (partite_list, inferred_tipo, inferred_esito)
    """
    if isinstance(partite_val, str):
        try:
            partite_val = json.loads(partite_val)
        except Exception:
            raise ValueError("partite must be valid JSON")

    if isinstance(partite_val, dict):
        partite_list = [partite_val]
    elif isinstance(partite_val, list):
        partite_list = partite_val
    elif partite_val is None:
        partite_list = []
    else:
        partite_list = [partite_val]

    inferred_tipo = _map_count_to_tipo_giocata(len(partite_list))

    outcomes = []
    for m in partite_list:
        outcomes.append(_detect_single_match_outcome(m))

    inferred_esito = None
    # outcomes already canonicalized to 'vinta'/'persa'/'in_progress' or None
    if any(o == "in_progress" for o in outcomes if o is not None):
        inferred_esito = "in_progress"
    elif any(o == "persa" for o in outcomes if o is not None):
        inferred_esito = "persa"
    else:
        if outcomes and len(outcomes) == len(partite_list) and all(o == "vinta" for o in outcomes):
            inferred_esito = "vinta"

    return partite_list, inferred_tipo, inferred_esito


def _build_bet_response(bet: Bet):
    """Return a dict shaped like BetRead for a Bet instance.

    Ensures `partite` is JSON-parsed when stored as string.
    """
    partite = bet.partite
    if isinstance(partite, str):
        partite = _safe_parse_json(partite)

    try:
        profit_amount = float(getattr(bet, "profitto", None)) if getattr(bet, "profitto", None) is not None else None
    except Exception:
        profit_amount = None
    if profit_amount is None:
        try:
            vincita = float(getattr(bet, "vincita", 0) or 0)
            importo = float(getattr(bet, "importo", 0) or 0)
            profit_amount = vincita - importo
        except Exception:
            profit_amount = 0.0

    try:
        importo_val = float(getattr(bet, "importo", 0) or 0)
        profit_pct = round((profit_amount / importo_val) * 100, 2) if importo_val else None
    except Exception:
        profit_pct = None

    return {
        "id": bet.id,
        "backroll_id": bet.backroll_id,
        "data": bet.data,
        "backroll_iniziale": bet.backroll_iniziale,
        "cassa": bet.cassa,
        "quota": bet.quota,
        "stake": bet.stake,
        "importo": bet.importo,
        "vincita": bet.vincita,
        "backroll_finale": bet.backroll_finale,
        "profitto_totale": bet.profitto_totale,
        "profitto": round(float(profit_amount or 0), 2),
        "esito": bet.esito,
        "partite": partite,
        "tipo_giocata": bet.tipo_giocata,
        "backroll_name": getattr(bet.backroll, "name", None),
        "link": bet.link,
        "note": bet.note,
        "created_at": bet.created_at,
        "profitto_percent": profit_pct,
    }


@router.get("/", response_model=List[BetRead], tags=["Bets"])
def list_bets(
    backroll_id: Optional[str] = Query(None, alias="backroll_id"),
    date: Optional[str] = Query(None, description="YYYY-MM-DD or prefix to filter bet.data"),
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD start date (inclusive)"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD end date (inclusive)"),
    status: Optional[str] = Query(None, description="played|won|lost|in_progress"),
    esito: Optional[str] = Query(None, description="match exact esito string"),
    tipo: Optional[str] = Query(None, description="Filter by tipo_giocata (alias: tipo)"),
    limit: int = Query(100, ge=1, le=1000, description="Max number of items to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    """Return bets filtered by optional backroll_id, date (prefix), status and esito."""
    q = db.query(Bet)
    if backroll_id:
        q = q.filter(Bet.backroll_id == backroll_id)

    if date:
        # filter by prefix (ISO date or date prefix)
        q = q.filter(Bet.data.like(f"{date}%"))

    # date range filtering (inclusive). Bets.data is stored as text; compare prefixes.
    # We accept YYYY-MM-DD strings and compare by prefix to include entire day.
    from_date = None
    to_date = None
    try:
        if date_from:
            # inclusive start
            from_date = datetime.strptime(date_from, "%Y-%m-%d").date()
        if date_to:
            to_date = datetime.strptime(date_to, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="date_from and date_to must be in YYYY-MM-DD format")

    if from_date and to_date and to_date < from_date:
        raise HTTPException(status_code=400, detail="date_to must be >= date_from")

    if from_date:
        q = q.filter(Bet.data >= from_date.isoformat())
    if to_date:
        # include whole day by comparing to the next day (exclusive)
        next_day = (to_date + timedelta(days=1)).isoformat()
        q = q.filter(Bet.data < next_day)

    if esito:
        q = q.filter(Bet.esito == esito)

    # filter by tipo_giocata (accept alias 'tipo')
    if tipo:
        # case-insensitive match
        q = q.filter(Bet.tipo_giocata.ilike(f"%{tipo}%"))

    # map high-level status to esito values
    if status:
        s = status.strip().lower()
        if s == "won":
            q = q.filter(Bet.esito.in_(["v", "win", "won", "vinto", "1"]))
        elif s == "lost":
            q = q.filter(Bet.esito.in_(["perso", "loss", "lost", "l", "0"]))
        elif s == "in_progress":
            # esito empty or null or pending
            q = q.filter((Bet.esito == None) | (Bet.esito == "") | (Bet.esito == "pending") | (Bet.esito == "in_progress") | (Bet.esito == "p"))
        elif s == "played":
            # not in in_progress
            q = q.filter(~((Bet.esito == None) | (Bet.esito == "") | (Bet.esito == "pending") | (Bet.esito == "in_progress") | (Bet.esito == "p")))

    # ensure bets are ordered by date descending, then by created_at descending for same date
    # join backroll to fetch its name in the same query
    bets = q.outerjoin(Backroll, Bet.backroll_id == Backroll.id).order_by(Bet.data.desc(), Bet.created_at.desc()).all()

    out = [_build_bet_response(b) for b in bets]
    return out


@router.post("/", response_model=BetRead, status_code=201, tags=["Bets"])
def create_bet(payload: BetCreate, db: Session = Depends(get_db)):
    """Create a new bet. If `id` is not provided it's generated.

    Validates that `backroll_id` exists when provided.
    """
    # validate backroll if provided
    if payload.backroll_id:
        br = db.query(Backroll).filter(Backroll.id == payload.backroll_id).first()
        if not br:
            raise HTTPException(status_code=400, detail="backroll_id not found")

    new_id = payload.id or str(uuid.uuid4())
    existing = db.query(Bet).filter(Bet.id == new_id).first()
    if existing:
        raise HTTPException(status_code=409, detail="Bet with this id already exists")

    # normalize partite and infer tipo_giocata/esito
    try:
        partite_list, inferred_tipo, inferred_esito = _normalize_partite_and_infer(payload.partite)
    except ValueError:
        raise HTTPException(status_code=400, detail="partite must be valid JSON")

    # default data to now if not provided
    data_val = payload.data or datetime.utcnow().isoformat()

    # determine final esito: prefer provided canonicalized payload, else inferred
    provided_esito = _canonicalize_esito(payload.esito) if getattr(payload, "esito", None) is not None else None
    final_esito = provided_esito or inferred_esito
    # if any match is 'persa' we force the whole bet to 'persa'
    if inferred_esito == "persa":
        final_esito = "persa"

    bet = Bet(
        id=new_id,
        backroll_id=payload.backroll_id,
        data=data_val,
        backroll_iniziale=payload.backroll_iniziale,
        cassa=payload.cassa,
        quota=payload.quota,
        stake=payload.stake,
        importo=payload.importo,
        vincita=payload.vincita,
        backroll_finale=payload.backroll_finale,
        profitto_totale=payload.profitto_totale,
        profitto=payload.profitto,
    # set esito
    esito=final_esito,
        partite=partite_list,
        # set tipo_giocata: prefer provided payload, else inferred
        tipo_giocata=(payload.tipo_giocata or inferred_tipo),
        link=payload.link,
        note=payload.note,
    )

    # enforce: if the bet is lost, vincita is 0 and profitto is -importo
    if getattr(bet, "esito", None) == "persa":
        try:
            imp = float(bet.importo or 0)
        except Exception:
            imp = 0.0
        bet.vincita = 0
        bet.profitto = -imp

    # enforce: if the bet is won, vincita = importo * quota and profitto = vincita - importo
    if getattr(bet, "esito", None) == "vinta":
        try:
            imp = float(bet.importo or 0)
        except Exception:
            imp = 0.0
        try:
            q = float(bet.quota or 0)
        except Exception:
            q = 0.0
        vincita_calc = round(imp * q, 2)
        bet.vincita = vincita_calc
        bet.profitto = round(vincita_calc - imp, 2)

    db.add(bet)
    db.commit()
    db.refresh(bet)

    # attach backroll_name if available
    backroll_name = getattr(bet.backroll, "name", None)

    return {
        "id": bet.id,
        "backroll_id": bet.backroll_id,
        "data": bet.data,
        "backroll_iniziale": bet.backroll_iniziale,
        "cassa": bet.cassa,
        "quota": bet.quota,
        "stake": bet.stake,
        "importo": bet.importo,
        "vincita": bet.vincita,
        "backroll_finale": bet.backroll_finale,
        "profitto_totale": bet.profitto_totale,
        "profitto": bet.profitto,
        "esito": bet.esito,
        "partite": bet.partite,
        "tipo_giocata": bet.tipo_giocata,
        "backroll_name": backroll_name,
        "link": bet.link,
        "note": bet.note,
        "created_at": bet.created_at,
        "profitto_percent": None,
    }



@router.delete("/{id}", status_code=204, tags=["Bets"])
def delete_bet(id: str, db: Session = Depends(get_db)):
    b = db.query(Bet).filter(Bet.id == id).first()
    if not b:
        raise HTTPException(status_code=404, detail="Bet not found")
    db.delete(b)
    db.commit()
    return None


@router.patch("/{id}", response_model=BetRead, tags=["Bets"])
def update_bet(id: str, payload: dict, db: Session = Depends(get_db)):
    """Patch/update a bet. Accepts partial fields in the payload.

    When `partite` is provided it will be normalized and the endpoint will re-infer
    `tipo_giocata` and `esito` using the same rules as create_bet.
    """
    b = db.query(Bet).filter(Bet.id == id).first()
    if not b:
        raise HTTPException(status_code=404, detail="Bet not found")

    # if backroll_id is being changed, validate it exists
    if "backroll_id" in payload and payload.get("backroll_id"):
        br = db.query(Backroll).filter(Backroll.id == payload.get("backroll_id")).first()
        if not br:
            raise HTTPException(status_code=400, detail="backroll_id not found")

    inferred_tipo = None
    inferred_esito = None
    if "partite" in payload:
        try:
            partite_list, inferred_tipo, inferred_esito = _normalize_partite_and_infer(payload.get("partite"))
        except ValueError:
            raise HTTPException(status_code=400, detail="partite must be valid JSON")
        payload["partite"] = partite_list

    # apply scalar updates
    updatable_fields = [
        "backroll_id",
        "data",
        "backroll_iniziale",
        "cassa",
        "quota",
        "stake",
        "importo",
        "vincita",
        "backroll_finale",
        "profitto_totale",
        "profitto",
        "esito",
        "partite",
        "tipo_giocata",
        "link",
        "note",
    ]

    for f in updatable_fields:
        if f in payload:
            setattr(b, f, payload.get(f))


    # if tipo_giocata wasn't explicitly provided but we inferred it, set it
    if ("tipo_giocata" not in payload or payload.get("tipo_giocata") in (None, "")) and inferred_tipo:
        b.tipo_giocata = inferred_tipo

    # if esito wasn't explicitly provided (or couldn't be canonicalized) but we inferred it, set it
    if ("esito" not in payload or _canonicalize_esito(payload.get("esito")) in (None, "")) and inferred_esito:
        b.esito = inferred_esito
    else:
        # if the user provided an esito, canonicalize and store it
        cand = _canonicalize_esito(payload.get("esito")) if "esito" in payload else None
        if cand:
            b.esito = cand

    # If any match in partite is 'persa' we force the whole bet to 'persa'
    try:
        if inferred_esito == "persa":
            b.esito = "persa"
    except Exception:
        pass

    # Always re-check the current `partite` (either existing or provided) and enforce overall esito
    try:
        current_partite = b.partite
        # normalize and infer from the current stored partite
        _, _ti, inferred_from_stored = _normalize_partite_and_infer(current_partite)
        if inferred_from_stored == "persa":
            b.esito = "persa"
    except Exception:
        # if normalization fails, ignore and keep existing esito
        pass

    # enforce: if the bet is lost, vincita is 0 and profitto is -importo
    if getattr(b, "esito", None) == "persa":
        try:
            imp = float(getattr(b, "importo", 0) or 0)
        except Exception:
            imp = 0.0
        b.vincita = 0
        b.profitto = -imp

    # enforce: if the bet is won, vincita = importo * quota and profitto = vincita - importo
    if getattr(b, "esito", None) == "vinta":
        try:
            imp = float(getattr(b, "importo", 0) or 0)
        except Exception:
            imp = 0.0
        try:
            q = float(getattr(b, "quota", 0) or 0)
        except Exception:
            q = 0.0
        vincita_calc = round(imp * q, 2)
        b.vincita = vincita_calc
        b.profitto = round(vincita_calc - imp, 2)

    db.add(b)
    db.commit()
    db.refresh(b)

    return _build_bet_response(b)
