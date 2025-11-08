
from __future__ import annotations
import logging, asyncio, httpx
import pandas as pd
from io import StringIO
from typing import Tuple
logger = logging.getLogger(__name__)

FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURES_CSV_OTHER_URL = "https://www.football-data.co.uk/new_league_fixtures.csv"

async def download_fixtures_pair(timeout: float = 30.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Scaricamento CSV fixtures in parallelo...")
    async with httpx.AsyncClient(timeout=timeout) as client:
        res_main, res_other = await asyncio.gather(
            client.get(FIXTURES_CSV_URL),
            client.get(FIXTURES_CSV_OTHER_URL)
        )
    res_main.raise_for_status(); res_other.raise_for_status()
    df_main = pd.read_csv(StringIO(res_main.text))
    df_other = pd.read_csv(StringIO(res_other.text))
    logger.info("Scaricati CSV: main=%d righe, other=%d righe", len(df_main), len(df_other))
    return df_main, df_other
