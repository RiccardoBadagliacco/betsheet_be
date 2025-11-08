# app/services/download_service.py
from __future__ import annotations
import asyncio
import httpx
import os
import io
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from app.constants.leagues import get_all_leagues
from app.services.csv_processor import filter_csv_bytes, split_recent_seasons, sha256_bytes

import logging
logger = logging.getLogger(__name__)

BASE_MAIN = "https://www.football-data.co.uk/mmz4281"
BASE_OTHER = "https://www.football-data.co.uk/new/{code}.csv"

def season_code_to_years(season: str) -> str:
    y1, y2 = int(season[:2]), int(season[2:])
    full1 = 1900 + y1 if y1 >= 50 else 2000 + y1
    return f"{full1}_{full1+1}"

def create_league_directory(league_code: str) -> Path:
    p = Path("leagues") / league_code
    p.mkdir(parents=True, exist_ok=True)
    return p

def generate_recent_seasons(num: int) -> List[str]:
    from datetime import datetime
    now = datetime.now()
    start = now.year if now.month >= 7 else now.year - 1
    return [f"{str(start-i)[-2:]}{str(start-i+1)[-2:]}" for i in range(num)]

async def _download_with_retry(client: httpx.AsyncClient, url: str, retries: int = 3, timeout: float = 20.0) -> bytes:
    last_exc = None
    for attempt in range(retries):
        try:
            r = await client.get(url, timeout=timeout)
            r.raise_for_status()
            logger.debug("Tentativo %d OK per %s", attempt+1, url)
            return r.content
        except Exception as e:
            last_exc = e
            logger.warning("Tentativo %d fallito (%s)", attempt+1, e)
            await asyncio.sleep(1.0 * (attempt + 1))
    raise last_exc

async def download_main_league(client: httpx.AsyncClient, league_code: str, seasons: List[str], *, overwrite_incomplete: bool) -> List[Tuple[str, str]]:
    logger.info("==> [%s] Avvio download per %d stagioni", league_code, len(seasons))

    out: List[Tuple[str, str]] = []
    league_dir = create_league_directory(league_code)
    for s in seasons:
        filename = f"{season_code_to_years(s)}.csv"
        path = league_dir / filename
        logger.debug("  -> Stagione %s, file %s", s, filename)
        if not overwrite_incomplete and path.exists():
            out.append((s, str(path)))
            continue
        url = f"{BASE_MAIN}/{s}/{league_code}.csv"
        try:
            raw = await _download_with_retry(client, url)
        except httpx.HTTPStatusError:
            continue
        filtered = filter_csv_bytes(raw)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            f.write(filtered)
        if path.exists() and sha256_bytes(filtered) == sha256_bytes(path.read_bytes()):
            tmp_path.unlink(missing_ok=True)
        else:
            os.replace(tmp_path, path)
        out.append((s, str(path)))
        await asyncio.sleep(0)
        logger.info("  ✅ Salvato: %s (%.1f KB)", path, os.path.getsize(path) / 1024)
    logger.info("==> [%s] completato (%d file)", league_code, len(out))

    return out

async def download_other_league(client: httpx.AsyncClient, league_code: str, *, n_seasons: int) -> List[Tuple[str, str]]:
    logger.info("==> [%s] Download CSV unico per lega 'other'", league_code)
    url = BASE_OTHER.format(code=league_code)
    raw = await _download_with_retry(client, url, timeout=25.0)
    filtered = filter_csv_bytes(raw)
    df = pd.read_csv(io.BytesIO(filtered))
    parts = split_recent_seasons(df, n_seasons)
    logger.info("  ✅ %d stagioni trovate nel dataset", len(parts))

    out: List[Tuple[str, str]] = []
    league_dir = create_league_directory(league_code)
    for season_key, df_season in parts.items():
        path = league_dir / f"{season_key}.csv"
        tmp_path = path.with_suffix('.tmp')
        df_season.to_csv(tmp_path, index=False)
        if path.exists() and sha256_bytes(tmp_path.read_bytes()) == sha256_bytes(path.read_bytes()):
            tmp_path.unlink(missing_ok=True)
        else:
            os.replace(tmp_path, path)
        out.append((season_key, str(path)))
    return out
