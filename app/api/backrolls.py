from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Backroll
from app.schemas.bet import BackrollRead
from app.schemas.bet import BackrollCreate
import uuid
from app.schemas.bet import BackrollUpdate
from sqlalchemy.orm import joinedload
from typing import List, Optional
from datetime import datetime, timedelta
from typing import Dict, Any

router = APIRouter()


@router.get("/", response_model=List[BackrollRead], tags=["Backrolls"])
def list_backrolls(id: Optional[str] = Query(None, alias="id"), db: Session = Depends(get_db)):
    """If `id` query param is provided, return a single-item list with that backroll.
    Otherwise return all backrolls with their bets. The response is always an array.
    """
    query = db.query(Backroll).options(joinedload(Backroll.bets))
    if id:
        query = query.filter(Backroll.id == id)
    brs = query.all()

    results = []
    for br in brs:
        # aggregate profit and stats
        total_profit = 0.0
        stats = {"Vinto": 0, "Giocate": 0, "Perso": 0, "In Progress": 0}

        # Collect bets and sort by datetime to compute running totals for history
        raw_bets = list(getattr(br, "bets", []) or [])
        parsed_bets = []
        print("Raw bets for backroll", br.name, ":", len(raw_bets))
        for b in raw_bets:
            # parse date if possible, fallback to created_at
            dt = None
            date_raw = getattr(b, "data", None)
            if date_raw:
                s = str(date_raw).strip()
                try:
                    dt = datetime.fromisoformat(s)
                except Exception:
                    try:
                        # common fallback format
                        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        dt = None
            
            # If we couldn't parse from data field, use created_at
            if dt is None and hasattr(b, 'created_at') and b.created_at:
                dt = b.created_at
            
            parsed_bets.append((b, dt))

        # sort bets by datetime (None go last), with created_at as secondary sort for same data
        parsed_bets.sort(key=lambda x: (x[1] is None, x[1] or datetime.min, getattr(x[0], 'created_at', None) or datetime.min))

        # use running total to compute per-day backroll
        bets_by_date = {}
        running_total = float(br.backroll or 0)

        for b, dt in parsed_bets:
            # profit value: prefer 'profitto', else vincita - importo
            try:
                profit_val = getattr(b, "profitto", None)
                profit_amount = float(profit_val) if profit_val is not None else None
            except Exception:
                profit_amount = None
            if profit_amount is None:
                try:
                    vincita = float(getattr(b, "vincita", 0) or 0)
                    importo = float(getattr(b, "importo", 0) or 0)
                    profit_amount = vincita - importo
                except Exception:
                    profit_amount = 0.0

            total_profit += float(profit_amount or 0)
            running_total += float(profit_amount or 0)

            # classify esito
            esito = (getattr(b, "esito", None) or "").strip().lower()
            if esito in ("v", "win", "won", "vinto", "1", "vinta"):
                stats["Vinto"] += 1
                stats["Giocate"] += 1
            elif esito in ("perso", "loss", "lost", "l", "0", "persa"):
                stats["Perso"] += 1
                stats["Giocate"] += 1
            elif esito in ("", None, "pending", "in_progress", "p", "in progress"):
                stats["In Progress"] += 1
            else:
                # treat as played
                stats["Giocate"] += 1

            # determine date_key from parsed dt or raw string
            date_key = None
            if dt:
                date_key = dt.date().isoformat()
            else:
                date_raw = getattr(b, "data", None)
                if date_raw:
                    ds = str(date_raw).strip()
                    date_key = ds[:10]

            if date_key:
                # store running_total as the backroll for that date (last bet of day will overwrite)
                bets_by_date[date_key] = running_total

        initial = float(br.backroll or 0)
        # compute win_rate: Vinto / Giocate * 100 (exclude In Progress from denominator)
        win_rate = None
        try:
            giocate = stats.get("Giocate", 0)
            vinti = stats.get("Vinto", 0)
            if giocate:
                win_rate = round((vinti / giocate) * 100, 2)
        except Exception:
            win_rate = None

        # build history: list of {date, backroll} using running_total captured per date
        history = []
        for dk in sorted(bets_by_date.keys()):
            br_val = bets_by_date[dk] or 0.0
            history.append({"date": dk, "backroll": round(float(br_val), 2)})
            
        # Build cumulative profit structure as requested: date array and values [[start,end],...]
        date_array = []
        values_array = []
        # iniziale: take the first total_backroll value (or 0 if missing)
        iniziale = history[0]["backroll"] if history else 0.0
        somma_cumulativa = 0.0
        for record in history:
            date_array.append(record["date"])
            start_value = iniziale + somma_cumulativa
            somma_cumulativa += record.get("backroll", 0.0) or 0.0
            end_value = iniziale + somma_cumulativa
            values_array.append([round(start_value, 2), round(end_value, 2)])
        
        # replace profit history with the requested structure
        profit_struct = {"date": date_array, "values": values_array}
        # determine current cassa and prefer last bet's backroll_finale as current_backroll
        current_cassa = None
        last_b = None
        try:
            if parsed_bets:
                last_b = parsed_bets[-1][0]
            else:
                last_b = None
            if last_b and getattr(last_b, "cassa", None) is not None:
                current_cassa = float(getattr(last_b, "cassa"))
        except Exception:
            current_cassa = None

        # compute current_backroll always from initial + total_profit (avoid using stored backroll_finale)
        try:
            current_backroll = round(initial + total_profit, 2)
        except Exception:
            current_backroll = round(total_profit, 2)

        # determine initial cassa (prefer the `cassa` defined on the backroll; fallback to br.backroll)
        try:
            initial_cassa = float(br.cassa) if getattr(br, "cassa", None) is not None else float(br.backroll or 0)
        except Exception:
            try:
                initial_cassa = float(br.backroll or 0)
            except Exception:
                initial_cassa = 0.0

        # If last bet has backroll_finale, compute profitto from that value (user expectation)
        last_br_final_f = None
        try:
            if last_b is not None:
                br_final = getattr(last_b, "backroll_finale", None)
                if br_final is not None and str(br_final).strip() != "":
                    try:
                        last_br_final_f = float(br_final)
                    except Exception:
                        last_br_final_f = None
        except Exception:
            last_br_final_f = None

        # If no backroll_finale in last bet, but br.cassa has been updated and differs from starting backroll,
        # consider br.cassa as the effective last backroll_finale (user expects profit = last_backroll - starting)
        try:
            if last_br_final_f is None and getattr(br, "cassa", None) is not None:
                try:
                    br_cassa_val = float(br.cassa)
                    starting_val = float(br.backroll or 0)
                    if br_cassa_val != starting_val:
                        last_br_final_f = br_cassa_val
                except Exception:
                    pass
        except Exception:
            pass

        try:
            # profit is defined as current_backroll - initial_cassa (user requested)
            profitto = round(current_backroll - initial_cassa, 2)
        except Exception:
            profitto = round(total_profit, 2)

        # profit percent relative to initial_cassa (if available)
        profit_percent = None
        try:
            if initial_cassa:
                profit_percent = round((profitto / initial_cassa) * 100, 2)
        except Exception:
            profit_percent = None
        
        # build bets list for response
        bets_list = []
        for b in getattr(br, "bets", []) or []:
            try:
                profit_val = getattr(b, "profitto", None)
                profit_amount = float(profit_val) if profit_val is not None else None
            except Exception:
                profit_amount = None
            if profit_amount is None:
                try:
                    vincita = float(getattr(b, "vincita", 0) or 0)
                    importo = float(getattr(b, "importo", 0) or 0)
                    profit_amount = vincita - importo
                except Exception:
                    profit_amount = 0.0

            try:
                importo_val = float(getattr(b, "importo", 0) or 0)
                profit_pct = round((profit_amount / importo_val) * 100, 2) if importo_val else None
            except Exception:
                profit_pct = None

            bets_list.append({
                "id": b.id,
                "data": b.data,
                "esito": b.esito,
                "importo": b.importo,
                "vincita": b.vincita,
                "backroll_finale": b.backroll_finale,
                "profitto": round(float(profit_amount or 0), 2),
                "profitto_percent": profit_pct,
            })

       

        result = {
            "id": br.id,
            "backroll": br.backroll,
            "cassa": br.cassa,
            "name": br.name,
            "bets": bets_list if bets_list else None,
            "current_backroll": current_backroll,
            "current_cassa": current_cassa if current_cassa is not None else br.cassa,
            "profitto": profitto,
            "profitto_percent": profit_percent,
            "win_rate": win_rate,
            "stats": stats,
            "history": history,
        }
        results.append(result)

    if id and not results:
        # asked for specific id but none found -> 404
        raise HTTPException(status_code=404, detail="Backroll not found")

    return results



@router.get("/stats", tags=["Backrolls"])
def backrolls_stats(
    include_per_backroll: bool = Query(True, description="Include per-backroll histories"),
    groupBy: str = Query("day", description="Raggruppa le statistiche per: day, week, month, year"),
    db: Session = Depends(get_db)
):
    """Return total history: for each date sum the backroll_finale of every backroll.

    Rule: if for a backroll there is no value for a certain date, use the previous day's value (carry-forward).
    If a backroll has no previous value, use its `backroll` (starting value).
    """
    # Load all backrolls with their bets
    brs = db.query(Backroll).options(joinedload(Backroll.bets)).all()

    # Helper: parse date string -> date iso
    def _parse_date_key(s) -> Optional[str]:
        if not s:
            return None
        try:
            dt = None
            ss = str(s).strip()
            try:
                dt = datetime.fromisoformat(ss)
            except Exception:
                try:
                    dt = datetime.strptime(ss, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    try:
                        dt = datetime.strptime(ss, "%Y-%m-%d")
                    except Exception:
                        dt = None
            if dt:
                return dt.date().isoformat()
        except Exception:
            return None
        return None

    # Helper: raggruppa la data secondo il groupBy richiesto
    def _group_date(date_str: str) -> str:
        from datetime import datetime
        if not date_str:
            return None
        dt = datetime.fromisoformat(date_str)
        if groupBy == "day":
            return dt.date().isoformat()
        elif groupBy == "week":
            # ISO week: YYYY-Www
            return f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"
        elif groupBy == "month":
            return dt.strftime("%Y-%m")
        elif groupBy == "year":
            return dt.strftime("%Y")
        else:
            return dt.date().isoformat()

    # collect all date keys present across bets
    all_dates = set()

    # per-backroll raw date->value map
    per_br_dates: Dict[str, Dict[str, float]] = {}

    for br in brs:
        starting = float(br.backroll or 0)
        per_map: Dict[str, float] = {}
        # build history using running total (like list_backrolls)
        raw_bets = list(getattr(br, "bets", []) or [])
        parsed_bets = []
        for b in raw_bets:
            dt = None
            date_raw = getattr(b, "data", None)
            if date_raw:
                s = str(date_raw).strip()
                try:
                    dt = datetime.fromisoformat(s)
                except Exception:
                    try:
                        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        dt = None
            
            # If we couldn't parse from data field, use created_at
            if dt is None and hasattr(b, 'created_at') and b.created_at:
                dt = b.created_at
            
            parsed_bets.append((b, dt))

        parsed_bets.sort(key=lambda x: (x[1] is None, x[1] or datetime.min, getattr(x[0], 'created_at', None) or datetime.min))

        running_total = float(br.backroll or 0)
        for b, dt in parsed_bets:
            # compute profit per bet
            try:
                profit_val = getattr(b, "profitto", None)
                profit_amount = float(profit_val) if profit_val is not None else None
            except Exception:
                profit_amount = None
            if profit_amount is None:
                try:
                    vincita = float(getattr(b, "vincita", 0) or 0)
                    importo = float(getattr(b, "importo", 0) or 0)
                    profit_amount = vincita - importo
                except Exception:
                    profit_amount = 0.0

            running_total += float(profit_amount or 0)

            # determine date_key
            date_key = None
            if dt:
                date_key = dt.date().isoformat()
            else:
                date_raw = getattr(b, "data", None)
                if date_raw:
                    ds = str(date_raw).strip()
                    date_key = ds[:10]

            if date_key:
                group_key = _group_date(date_key)
                per_map[group_key] = running_total
                all_dates.add(group_key)

        per_br_dates[br.id] = {"name": br.name, "starting": starting, "dates": per_map}

    if not all_dates:
        return {"total": {"backroll": []}, "per_backroll": {} if include_per_backroll else None}

    # build full sorted date list
    sorted_dates = sorted(all_dates)
    full_dates = sorted_dates
    # Solo per day si può interpolare le date mancanti, per week/month/year si usano solo le chiavi presenti
    if groupBy == "day" and sorted_dates:
        min_date = datetime.fromisoformat(sorted_dates[0]).date()
        max_date = datetime.fromisoformat(sorted_dates[-1]).date()
        span = (max_date - min_date).days
        full_dates = [(min_date + timedelta(days=i)).isoformat() for i in range(span + 1)]
    # altrimenti, per week/month/year, usiamo solo le date raggruppate trovate

    # fill per-backroll series by carry-forward, starting from starting value
    per_br_filled: Dict[str, Dict[str, float]] = {}
    for br_id, info in per_br_dates.items():
        filled = {}
        # contribution is 0 before the first bet
        prev = 0.0
        dates_map = info.get("dates", {})
        for dk in full_dates:
            if dk in dates_map:
                prev = dates_map[dk]
            filled[dk] = prev
        per_br_filled[br_id] = {"name": info.get("name"), "series": filled}

    # compute total history by summing per-backroll filled values per date
    history = []
    for dk in full_dates:
        total = 0.0
        for br_id, info in per_br_filled.items():
            total += float(info["series"].get(dk, 0) or 0)
        history.append({"date": dk, "total_backroll": round(total, 2)})

    # Build profit pairs directly from the total backroll history:
    # for each date return [previous_total_backroll, current_total_backroll]
    # compute profit_by_date by summing bet-level profits per date
    from collections import defaultdict
    profit_by_date = defaultdict(float)
    for br in brs:
        for b in getattr(br, "bets", []) or []:
            # compute profit for the bet (prefer explicit 'profitto')
            try:
                profit_val = getattr(b, "profitto", None)
                profit_amount = float(profit_val) if profit_val is not None else None
            except Exception:
                profit_amount = None
            if profit_amount is None:
                try:
                    vincita = float(getattr(b, "vincita", 0) or 0)
                    importo = float(getattr(b, "importo", 0) or 0)
                    profit_amount = vincita - importo
                except Exception:
                    profit_amount = 0.0

            date_key = _parse_date_key(getattr(b, "data", None))
            if date_key:
                group_key = _group_date(date_key)
                profit_by_date[group_key] += float(profit_amount or 0.0)

    date_array = []
    values_array = []
    for rec in history:
        dk = rec.get("date")
        cur_total = float(rec.get("total_backroll", 0.0) or 0.0)
        daily_profit = float(profit_by_date.get(dk, 0.0) or 0.0)
        prev_total = cur_total - daily_profit
        date_array.append(dk)
        values_array.append([round(prev_total, 2), round(cur_total, 2)])

    profit_struct = {"date": date_array, "values": values_array}

    result: Dict[str, Any] = {"total": {"backroll": history, "profit": profit_struct}}
    if include_per_backroll:
        # include per-backroll compact history
        pb = {}
        for br_id, info in per_br_filled.items():
            # compact series to list of {date, value}
            series = [{"date": d, "backroll": round(v, 2)} for d, v in info["series"].items()]
            pb[br_id] = {"name": info["name"], "series": series}
        result["per_backroll"] = pb
    else:
        result["per_backroll"] = None

    return result


@router.post("/", response_model=BackrollRead, status_code=201, tags=["Backrolls"])
def create_backroll(payload: BackrollCreate, db: Session = Depends(get_db)):
    """Create a new backroll. If `id` is not provided it's generated.
    """
    new_id = payload.id or str(uuid.uuid4())
    exists = db.query(Backroll).filter(Backroll.id == new_id).first()
    if exists:
        raise HTTPException(status_code=409, detail="Backroll with this id already exists")

    br = Backroll(id=new_id, backroll=payload.backroll, cassa=payload.cassa, name=payload.name)
    db.add(br)
    db.commit()
    db.refresh(br)

    return {
        "id": br.id,
        "backroll": br.backroll,
        "cassa": br.cassa,
        "name": br.name,
        "bets": None,
        "current_backroll": br.backroll,
        "current_cassa": br.cassa,
        "profitto": 0.0,
        "profitto_percent": None,
        "win_rate": None,
        "stats": {"Vinto": 0, "Giocate": 0, "Perso": 0, "In Progress": 0},
        "history": [],
    }



@router.patch("/{id}", response_model=BackrollRead, tags=["Backrolls"])
def update_backroll(id: str, payload: BackrollUpdate, db: Session = Depends(get_db)):
    br = db.query(Backroll).filter(Backroll.id == id).first()
    if not br:
        raise HTTPException(status_code=404, detail="Backroll not found")

    if payload.backroll is not None:
        br.backroll = payload.backroll
    if payload.cassa is not None:
        br.cassa = payload.cassa
    if payload.name is not None:
        br.name = payload.name

    db.add(br)
    db.commit()
    db.refresh(br)

    # return same structure as list endpoint for convenience
    return {
        "id": br.id,
        "backroll": br.backroll,
        "cassa": br.cassa,
        "name": br.name,
        "bets": None,
        "current_backroll": br.backroll,
        "current_cassa": br.cassa,
        "profitto": 0.0,
        "profitto_percent": None,
        "win_rate": None,
        "stats": {"Vinto": 0, "Giocate": 0, "Perso": 0, "In Progress": 0},
        "history": [],
    }


@router.delete("/{id}", status_code=204, tags=["Backrolls"])
def delete_backroll(id: str, db: Session = Depends(get_db)):
    br = db.query(Backroll).filter(Backroll.id == id).first()
    if not br:
        raise HTTPException(status_code=404, detail="Backroll not found")
    # cascade delete should remove bets thanks to relationship cascade
    db.delete(br)
    db.commit()
    return None



@router.get("/v2/stats", tags=["Backrolls"])
def backrolls_stats_v2(
    groupBy: str = Query("day", description="Raggruppa le statistiche per: day, week, month, year"),
    type: str = Query("backroll", description="Tipo di dati: 'total' (solo date e total), 'backroll' (date + backrolls), 'profit' (profit bars)"),
    db: Session = Depends(get_db)
):
    """Return stats in Recharts-compatible format based on type:
    - type=total: [{date, total}, ...]
    - type=backroll: [{date, backroll_1, backroll_2, ...}, ...]  
    - type=profit: [{date, profit, positive, positiveBase, positiveHeight, negativeBase, negativeHeight, cumulativeTotal}, ...]
    """
    # Load all backrolls with their bets
    brs = db.query(Backroll).options(joinedload(Backroll.bets)).all()

    # Helper: parse date string -> date iso
    def _parse_date_key(s) -> Optional[str]:
        if not s:
            return None
        try:
            dt = None
            ss = str(s).strip()
            try:
                dt = datetime.fromisoformat(ss)
            except Exception:
                try:
                    dt = datetime.strptime(ss, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    try:
                        dt = datetime.strptime(ss, "%Y-%m-%d")
                    except Exception:
                        dt = None
            if dt:
                return dt.date().isoformat()
        except Exception:
            return None
        return None

    # Helper: raggruppa la data secondo il groupBy richiesto
    def _group_date(date_str: str) -> str:
        from datetime import datetime
        if not date_str:
            return None
        dt = datetime.fromisoformat(date_str)
        if groupBy == "day":
            return dt.date().isoformat()
        elif groupBy == "week":
            # ISO week: YYYY-Www
            return f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"
        elif groupBy == "month":
            return dt.strftime("%Y-%m")
        elif groupBy == "year":
            return dt.strftime("%Y")
        else:
            return dt.date().isoformat()

    # collect all date keys present across bets
    all_dates = set()

    # per-backroll raw date->value map
    per_br_dates: Dict[str, Dict[str, float]] = {}
    backroll_names = {}

    for br in brs:
        starting = float(br.backroll or 0)
        per_map: Dict[str, float] = {}
        backroll_names[br.id] = br.name or f"Backroll {br.id[:8]}"
        
        # build history using running total (like list_backrolls)
        raw_bets = list(getattr(br, "bets", []) or [])
        parsed_bets = []
        for b in raw_bets:
            dt = None
            date_raw = getattr(b, "data", None)
            if date_raw:
                s = str(date_raw).strip()
                try:
                    dt = datetime.fromisoformat(s)
                except Exception:
                    try:
                        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        dt = None
            
            # If we couldn't parse from data field, use created_at
            if dt is None and hasattr(b, 'created_at') and b.created_at:
                dt = b.created_at
            
            parsed_bets.append((b, dt))

        parsed_bets.sort(key=lambda x: (x[1] is None, x[1] or datetime.min, getattr(x[0], 'created_at', None) or datetime.min))

        running_total = float(br.backroll or 0)
        for b, dt in parsed_bets:
            # compute profit per bet
            try:
                profit_val = getattr(b, "profitto", None)
                profit_amount = float(profit_val) if profit_val is not None else None
            except Exception:
                profit_amount = None
            if profit_amount is None:
                try:
                    vincita = float(getattr(b, "vincita", 0) or 0)
                    importo = float(getattr(b, "importo", 0) or 0)
                    profit_amount = vincita - importo
                except Exception:
                    profit_amount = 0.0

            running_total = round(running_total + float(profit_amount or 0), 2)

            # determine date_key
            date_key = None
            if dt:
                date_key = dt.date().isoformat()
            else:
                date_raw = getattr(b, "data", None)
                if date_raw:
                    ds = str(date_raw).strip()
                    date_key = ds[:10]

            if date_key:
                group_key = _group_date(date_key)
                per_map[group_key] = round(running_total, 2)
                all_dates.add(group_key)

        per_br_dates[br.id] = {"name": br.name, "starting": round(starting, 2), "dates": per_map}

    if not all_dates:
        return []

    # build full sorted date list
    sorted_dates = sorted(all_dates)
    full_dates = sorted_dates
    # Solo per day si può interpolare le date mancanti, per week/month/year si usano solo le chiavi presenti
    if groupBy == "day" and sorted_dates:
        min_date = datetime.fromisoformat(sorted_dates[0]).date()
        max_date = datetime.fromisoformat(sorted_dates[-1]).date()
        span = (max_date - min_date).days
        full_dates = [(min_date + timedelta(days=i)).isoformat() for i in range(span + 1)]

    # fill per-backroll series by carry-forward, starting from starting value
    per_br_filled: Dict[str, Dict[str, float]] = {}
    for br_id, info in per_br_dates.items():
        filled = {}
        # contribution starts from the starting value
        prev = round(info.get("starting", 0.0), 2)
        dates_map = info.get("dates", {})
        for dk in full_dates:
            if dk in dates_map:
                prev = round(dates_map[dk], 2)
            filled[dk] = prev
        per_br_filled[br_id] = filled

    # Build response based on type parameter
    if type == "total":
        # Return only date and total
        result = []
        for dk in full_dates:
            total = 0.0
            for br_id, series in per_br_filled.items():
                total += float(series.get(dk, 0) or 0)
            result.append({
                "date": dk,
                "total": round(total, 2)
            })
        return result
    
    elif type == "profit":
        # Calculate daily profits for profit bars with start/end logic
        from collections import defaultdict
        profit_by_date = defaultdict(float)
        
        # Collect daily profits from bets
        for br in brs:
            for b in getattr(br, "bets", []) or []:
                try:
                    profit_val = getattr(b, "profitto", None)
                    profit_amount = float(profit_val) if profit_val is not None else None
                except Exception:
                    profit_amount = None
                if profit_amount is None:
                    try:
                        vincita = float(getattr(b, "vincita", 0) or 0)
                        importo = float(getattr(b, "importo", 0) or 0)
                        profit_amount = vincita - importo
                    except Exception:
                        profit_amount = 0.0

                date_key = _parse_date_key(getattr(b, "data", None))
                if date_key:
                    group_key = _group_date(date_key)
                    profit_by_date[group_key] = round(profit_by_date[group_key] + float(profit_amount or 0.0), 2)
        
        # Build profit structure with start/end logic like frontend
        result = []
        
        # Calculate initial total (sum of all starting backrolls)
        initial_total = 0.0
        for br in brs:
            initial_total = round(initial_total + float(br.backroll or 0), 2)
        
        # Track cumulative profit (starting from 0)
        cumulative_profit = 0.0
        
        for i, dk in enumerate(full_dates):
            daily_profit = round(float(profit_by_date.get(dk, 0.0) or 0.0), 2)
            
            if i == 0:
                # First entry: show initial backroll total as starting point
                cumulative_profit = round(cumulative_profit + daily_profit, 2)
                current_total = round(initial_total + cumulative_profit, 2)
                result.append({
                    "date": dk,
                    "start": round(initial_total, 2),
                    "end": current_total,
                    "value": round(initial_total, 2)  # Initial backroll value
                })
            else:
                # Subsequent entries: delta logic for profit changes
                start_total = round(initial_total + cumulative_profit, 2)
                cumulative_profit = round(cumulative_profit + daily_profit, 2)
                end_total = round(initial_total + cumulative_profit, 2)
                
                result.append({
                    "date": dk,
                    "delta": daily_profit,
                    "start": start_total,
                    "end": end_total
                })
        
        return result
    
    else:  # type == "backroll" or default
        # Return date + individual backrolls (original behavior)
        result = []
        for dk in full_dates:
            entry = {"date": dk}
            
            # Add each backroll as a separate field
            for br_id, series in per_br_filled.items():
                br_value = round(float(series.get(dk, 0) or 0), 2)
                # Use backroll name as key, fallback to ID if name is empty
                key = backroll_names.get(br_id) or f"backroll_{br_id[:8]}"
                # Sanitize key for JSON compatibility (remove spaces and special chars)
                key = key.replace(" ", "_").replace("-", "_").replace(".", "_")
                entry[key] = br_value
            
            result.append(entry)
        return result


@router.get("/total", tags=["Backrolls"])
def total_backroll(db: Session = Depends(get_db)):
    """Return the total current backroll considering all backrolls (initial + bets profit)."""
    brs = db.query(Backroll).options(joinedload(Backroll.bets)).all()
    grand_total = 0.0
    grand_profit = 0.0
    count = 0
    for br in brs:
        count += 1
        total_profit = 0.0
        for b in getattr(br, "bets", []) or []:
            try:
                p = float(getattr(b, "profitto", 0) or 0)
            except Exception:
                p = 0.0
            total_profit += p
        grand_profit += total_profit
        initial = float(br.backroll or 0)
        current = initial + total_profit
        grand_total += current

    grand_total = round(grand_total, 2)
    # round profit too
    grand_profit = round(grand_profit, 2)
    return {"total_backroll": grand_total, "total_profit": grand_profit, "count": count}
