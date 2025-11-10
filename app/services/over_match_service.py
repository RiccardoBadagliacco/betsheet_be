import json
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

from zoneinfo import ZoneInfo

DATA_FILE = "data/all_predictions.json"


def _add_minutes_to_datetime(dt: datetime, minutes: int) -> datetime:
    return dt + timedelta(minutes=minutes)


def _format_time(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def get_today_over_groups(
    min_confidence: float = 0.8,
    source_tz: str = "UTC",
    target_tz: str = "Europe/Rome",
) -> Dict[str, List[str]]:
    """Read predictions and return a dict grouped by shifted match_time (+50 min)

    Times are converted from `source_tz` to `target_tz` (Italy by default).
    Grouping is performed on the match local time in `target_tz`, then each
    group key is increased by 50 minutes and returned as HH:MM strings in
    the `target_tz` timezone.
    """
    tz_src = ZoneInfo(source_tz)
    tz_tgt = ZoneInfo(target_tz)

    # today's date in target timezone
    today_dt = datetime.now(tz=tz_tgt)
    today_str = today_dt.date().isoformat()

    with open(DATA_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    fixtures = data.get("fixtures", [])

    groups: Dict[str, List[str]] = defaultdict(list)

    for f in fixtures:
        try:
            # parse match date/time
            md = f.get("match_date")
            mt = f.get("match_time")
            if not md or not mt:
                continue

            # build naive datetime from strings
            try:
                dt_naive = datetime.strptime(f"{md} {mt}", "%Y-%m-%d %H:%M")
            except Exception:
                continue

            # interpret in source tz then convert to target tz
            dt_src = dt_naive.replace(tzinfo=tz_src)
            dt_tgt = dt_src.astimezone(tz_tgt)

            # only consider fixtures that are today in target tz
            if dt_tgt.date().isoformat() != today_str:
                continue

            # find Over 1.5 confidence (recommendations preferred)
            recs = f.get("recommendations", {}).get("over_under", [])
            over15_conf: Optional[float] = None
            for r in recs:
                market = r.get("market", "")
                if market and market.replace(" ", "").lower() == "over1.5":
                    over15_conf = r.get("confidence")
                    break
                if market == "Over 1.5":
                    over15_conf = r.get("confidence")
                    break

            if over15_conf is None:
                # fallback to predictions
                p = f.get("predictions", {}).get("over_under", {}).get("p_over_1_5")
                if isinstance(p, (float, int)):
                    over15_conf = float(p)

            if over15_conf is None or float(over15_conf) < float(min_confidence):
                continue

            # grouping key: local time in target tz
            local_time = dt_tgt
            grouped_key = _format_time(local_time)

            match_label = f"{f.get('home_team')} vs {f.get('away_team')}"
            groups[grouped_key].append(match_label)
        except Exception:
            continue

    # shift each key by +50 minutes and produce final dict
    shifted: Dict[str, List[str]] = {}
    for key, matches in groups.items():
        try:
            dt = datetime.strptime(key, "%H:%M")
            dt = _add_minutes_to_datetime(dt, 50)
            new_key = dt.strftime("%H:%M")
        except Exception:
            new_key = key
        shifted[new_key] = matches

    return dict(sorted(shifted.items()))
