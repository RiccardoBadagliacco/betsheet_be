#!/usr/bin/env python3
# ==============================================================
# analyze_thresholds.py
# Analisi soglie ottimali + generazione JSON
# ==============================================================

import pandas as pd
from pathlib import Path
import json

LOG_FILE = Path("threshold_analysis_log.csv")
OUTPUT_JSON = Path("optimal_thresholds.json")

def analyze_thresholds(bucket_size: int = 5, min_samples: int = 20, alpha: float = 0.7):
    """
    Analizza i log e seleziona soglie ottimali per mercato bilanciando accuracy e coverage.
    
    :param bucket_size: ampiezza dei bucket (es. 5%)
    :param min_samples: numero minimo di esempi per considerare un bucket valido
    :param alpha: peso dell'accuracy nel calcolo dello score finale (0–1)
    """
    if not LOG_FILE.exists():
        print(f"❌ Nessun file di log trovato: {LOG_FILE}")
        return

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        print(f"❌ Il file {LOG_FILE} è vuoto.")
        return

    total_per_market = df.groupby('market')['correct'].count()

    # Raggruppa per bucket e calcola accuracy
    df['bucket'] = (df['confidence'] // bucket_size) * bucket_size
    grouped = df.groupby(['market', 'bucket']).agg(
        picks=('correct', 'count'),
        accuracy=('correct', 'mean')
    ).reset_index()

    print(f"\n📊 ANALISI SOGLIE PER MERCATO (bucket {bucket_size}%) — α={alpha}")
    print("-" * 70)

    optimal_thresholds = {}

    for market in grouped['market'].unique():
        mkt = grouped[grouped['market'] == market]
        mkt = mkt[mkt['picks'] >= min_samples]
        if mkt.empty:
            continue

        total = total_per_market[market]
        print(f"\nMercato: {market}")
        print(f"{'Bucket':>8} | {'Picks':>6} | {'Acc':>7} | {'Cov':>7} | {'Score':>7}")
        print("-" * 40)

        best_row = None
        for _, row in mkt.iterrows():
            cov = (row['picks'] / total) * 100
            score = alpha * (row['accuracy'] * 100) + (1 - alpha) * cov
            print(f"{int(row['bucket']):>8} | {int(row['picks']):>6} | {row['accuracy']*100:6.2f}% | {cov:6.2f}% | {score:6.3f}")
            row_data = {**row, 'cov': cov, 'score': score}
            if best_row is None or score > best_row['score']:
                best_row = row_data

        if best_row is not None:
            chosen_thr = int(best_row['bucket'])
            optimal_thresholds[market] = chosen_thr
            print(f"--> Soglia scelta: {chosen_thr}% (Acc: {best_row['accuracy']*100:.2f}%, Cov: {best_row['cov']:.2f}%)")

    # Salvataggio in JSON
    if optimal_thresholds:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(optimal_thresholds, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Soglie ottimali salvate in: {OUTPUT_JSON}")
    else:
        print("\n⚠️ Nessuna soglia ottimale trovata — JSON non generato.")


if __name__ == "__main__":
    analyze_thresholds()
