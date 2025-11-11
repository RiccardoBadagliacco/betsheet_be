# app/services/csv_processor.py
from __future__ import annotations
import io
import hashlib
from typing import Dict, List
import pandas as pd

import logging
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = [
    'Date', 'Time', 'HomeTeam', 'AwayTeam', 'HTHG', 'HTAG', 'FTHG', 'FTAG', 'Season',
    'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5'
]

COLUMN_MAPPINGS: Dict[str, List[str]] = {
    'AvgH': ['AvgH', 'BbAvH','AvgCH'],
    'AvgD': ['AvgD', 'BbAvD','AvgCD'],
    'AvgA': ['AvgA', 'BbAvA','AvgCA'],
    'Avg>2.5': ['Avg>2.5','BbAv>2.5'],
    'Avg<2.5': ['Avg<2.5','BbAv<2.5'],
    'HomeTeam': ['HomeTeam','Home'],
    'AwayTeam': ['AwayTeam','Away'],
    'FTHG': ['FTHG','HG'],
    'FTAG': ['FTAG','AG'],
    'Season': ['Season'],
    'Time': ['Time'],
    'HTHG': ['HTHG'],
    'HTAG': ['HTAG'],
}

USECOLS: List[str] = sorted({alt for v in COLUMN_MAPPINGS.values() for alt in v})

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def filter_csv_bytes(csv_bytes: bytes) -> bytes:
    logger.debug("Inizio filtraggio CSV (size: %.1f KB)", len(csv_bytes) / 1024)
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes), usecols=lambda c: True, low_memory=False)
    except Exception as e:
        logger.error("Errore durante la lettura CSV: %s", e)
        raise
    available = set(df.columns)
    out = {}
    for target in REQUIRED_COLUMNS:
        if target in available:
            out[target] = df[target]
            continue
        for alt in COLUMN_MAPPINGS.get(target, []):
            if alt in available:
                out[target] = df[alt]
                break
        else:
            out[target] = pd.Series([None] * len(df))
    filtered = pd.DataFrame(out)[REQUIRED_COLUMNS]

    return filtered.to_csv(index=False).encode('utf-8')

def split_recent_seasons(df: pd.DataFrame, n_seasons: int):
    if 'Season' not in df.columns:
        return {}
    unique = (df['Season'].astype(str).str.strip().replace({'\\\\': '/'}, regex=True))
    seasons = list(pd.Series(unique).drop_duplicates().tail(n_seasons))
    out = {}
    for s in seasons:
        s_key = s.replace('/', '_')
        out[s_key] = df[df['Season'].astype(str) == s]
    return out
