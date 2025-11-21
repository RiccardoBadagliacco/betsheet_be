    #!/usr/bin/env python3
"""Utility per stampare tutte le colonne e il numero di righe di un file Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mostra lista completa colonne e numero di righe per un Parquet."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Percorso del file .parquet da analizzare",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path = args.path.expanduser().resolve()

    if not parquet_path.exists():
        raise SystemExit(f"File non trovato: {parquet_path}")

    # carica il parquet (necessita di pandas + pyarrow/fastparquet)
    df = pd.read_parquet(parquet_path)

    print(f"File: {parquet_path}")
    print(f"Numero di colonne: {len(df.columns)}")
    print(f"Numero di righe: {len(df)}")

    print("\nPrime 5 righe:")
    print(df.head())

    print("\nColonne:")
    print(df.columns)


if __name__ == "__main__":
    main()
