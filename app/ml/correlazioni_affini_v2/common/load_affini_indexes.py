from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "correlazioni_affini_v2" / "data"

SLIM_PATH = DATA_DIR / "step4b_affini_index_slim_v2.parquet"
WIDE_PATH = DATA_DIR / "step4a_affini_index_wide_v2.parquet"


def load_affini_indexes():
    """Carica gli indici affini usati dal soft engine"""
    slim = pd.read_parquet(SLIM_PATH)
    wide = pd.read_parquet(WIDE_PATH)
    return slim, wide