#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP7 — AFFINI SOFT ENGINE (single match) V2
--------------------------------------------

A partire da:
  - step4b_affini_index_slim_v2.parquet  (indice compatto con cluster + feature chiave)
  - step4a_affini_index_wide_v2.parquet  (master con risultati reali)

Per un match target:
  1. Seleziona candidati "hard" (cluster + range su elo/lambda/market_sharpness)
  2. Calcola distanza multivariata su feature chiave
  3. Converte le distanze in pesi continui (kernel esponenziale)
  4. Calcola probabilità soft:
       - 1X2: p1, px, p2
       - OU1.5: pover15, punder15
       - OU2.5: pover25, punder25
  5. Salva JSON con:
       - info match target
       - stats affini
       - probabilità soft finali
       - lista affini (con distanza e pesi)
"""

import json
from pathlib import Path
import sys

# path root progetto
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.correlazioni_affini_v2.common.soft_engine_api_v2 import (
    load_affini_indexes,
    run_soft_engine_from_indexes,
)

BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
OUTPUT_DIR = AFFINI_DIR / "affini_results_soft"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="STEP7 — Affini Soft Engine (single match) V2"
    )
    ap.add_argument(
        "--match_id",
        type=str,
        default="latest",
        help="match_id target, oppure 'latest' per l'ultimo in ordine di data",
    )
    ap.add_argument(
        "--top_n",
        type=int,
        default=80,
        help="numero massimo di affini soft da usare",
    )
    ap.add_argument(
        "--min_neighbors",
        type=int,
        default=30,
        help="minimo di vicini richiesti prima di allargare i filtri",
    )

    args = ap.parse_args()

    print("============================================================")
    print("🚀 STEP7 — AFFINI SOFT ENGINE (single match) V2")
    print("============================================================")

    # 1) Carico indici
    slim, wide = load_affini_indexes()
    print(f"📥 SLIM index shape: {slim.shape}")
    print(f"📥 WIDE index shape: {wide.shape}")

    # 2) Scelgo il match target
    if args.match_id == "latest":
        t0 = slim.sort_values("date").iloc[-1]
        match_id = str(t0["match_id"])
    else:
        match_id = args.match_id
        rows = slim.loc[slim["match_id"] == match_id]
        if rows.empty:
            raise RuntimeError(f"❌ match_id {match_id} NON trovato nello SLIM index!")
        t0 = rows.iloc[0]

    print(
        f"\n🎯 Target match: {t0['match_id']} | "
        f"{t0.get('home_team')} - {t0.get('away_team')} | "
        f"data={t0.get('date')}"
    )

    # 3) Eseguo SOFT ENGINE (con neighbors per debug/JSON)
    res = run_soft_engine_from_indexes(
        target_match_id=match_id,
        slim_index=slim,
        wide_index=wide,
        top_n=args.top_n,
        min_neighbors=args.min_neighbors,
        return_neighbors=True,
    )

    status = res.get("status")
    if status != "ok":
        print("\n⚠️ SOFT ENGINE NON OK")
        print(f"   status = {status}")
        print(f"   reason = {res.get('reason')}")
        # salvo comunque un JSON minimale di errore
        out_err = {
            "status": status,
            "reason": res.get("reason"),
            "target_match": {
                "match_id": match_id,
                "date": str(t0.get("date")),
                "home_team": t0.get("home_team"),
                "away_team": t0.get("away_team"),
            },
        }
        out_path = OUTPUT_DIR / f"affini_soft_{match_id}.json"
        with open(out_path, "w") as f:
            json.dump(out_err, f, indent=2, default=str)
        print(f"💾 Salvato JSON errore → {out_path}")
        print("============================================================")
        return

    soft = res["soft_probs"]
    stats = res["affini_stats"]
    clusters = res["clusters"]

    # 4) Log riassuntivo
    print("\n🎯 PROBABILITÀ SOFT (affini):")
    print(
        f"   1X2  → P1={soft['p1']:.3f}, "
        f"PX={soft['px']:.3f}, "
        f"P2={soft['p2']:.3f}, "
        f"sum={soft['p1'] + soft['px'] + soft['p2']:.3f}"
    )
    print(
        f"   O1.5 → P(Over)={soft['over15']:.3f}, "
        f"P(Under)={soft['under15']:.3f}"
    )
    print(
        f"   O2.5 → P(Over)={soft['over25']:.3f}, "
        f"P(Under)={soft['under25']:.3f}"
    )

    print("\n📊 STATISTICHE AFFINI:")
    print(f"   n_affini_soft  = {stats.get('n_affini_soft')}")
    print(f"   avg_distance   = {stats.get('avg_distance'):.4f}")
    print(f"   min_distance   = {stats.get('min_distance'):.4f}")
    print(f"   max_distance   = {stats.get('max_distance'):.4f}")
    print(f"   cluster_1x2    = {clusters.get('cluster_1x2')}")
    print(f"   cluster_ou25   = {clusters.get('cluster_ou25')}")
    print(f"   cluster_ou15   = {clusters.get('cluster_ou15')}")
    print(f"   config_used    = {res.get('config_used')}")

    # 5) Costruisco JSON di output stile vecchio step7
    out = {
        "target_match": res["target"],
        "clusters": clusters,
        "soft_probs": soft,
        "affini_stats": stats,
        "config_used": res.get("config_used"),
        "affini_list": res.get("neighbors", []),
    }

    out_path = OUTPUT_DIR / f"affini_soft_{match_id}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n💾 Salvato risultato SOFT → {out_path}")
    print("🏁 STEP7 SOFT (single) COMPLETATO!")
    print("============================================================")


if __name__ == "__main__":
    main()