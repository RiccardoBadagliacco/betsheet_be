#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RUN PROFETA — PIPELINE UFFICIALE (WEEKLY)

Questa pipeline va eseguita:
- DOPO ogni aggiornamento del database
- QUANDO entrano nuovi match storici
- QUANDO entrano nuove fixture future

OBIETTIVO
---------
Costruire la conoscenza OFFLINE di Profeta.

Produce:
- probabilità aggiornate per match storici e fixture future
- stati di partita data-driven (solo storico)
- profili statistici stati × mercati (solo storico)

⚠️ NON genera alert runtime
⚠️ NON prende decisioni di betting
⚠️ NON usa quote bookmaker

Gli alert vengono generati SOLO a runtime dall’Alert Engine.
"""

import subprocess
import sys
from pathlib import Path

# ============================================================
# PATH & PYTHON
# ============================================================

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: str):
    print("\n" + "=" * 80)
    print(f"▶️  ESEGUO: {cmd}")
    print("=" * 80)
    res = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    if res.returncode != 0:
        print("❌ ERRORE — pipeline interrotta")
        sys.exit(1)


def main():
    data_dir = ROOT / "data"
    dc_file = data_dir / "step1b_profeta_dc_params.json"

    # =========================================================
    # FASE A — MODELLO (STATISTICA PURA)
    # =========================================================
    """
    In questa fase Profeta:
    - legge il DB aggiornato
    - ricostruisce il dataset
    - ri-allena il modello goal-based
    - produce la fotografia probabilistica
      per match storici + fixture future
    """

    # 1) Dataset base (storico + fixture)
    run(f"{PY} step0_profeta.py")

    # 2) Feature di forma
    #    (calcolate SOLO su storico,
    #     propagate alle fixture quando possibile)
    run(f"{PY} step0b_profeta_form.py")

    # 3) Training modello goal-based
    run(f"{PY} step1_profeta_train.py")

    # 4) Prediction BASE (rho = 0)
    if dc_file.exists():
        print("🧹 Rimuovo vecchio rho Dixon–Coles")
        dc_file.unlink()

    run(f"{PY} step3_profeta_batch.py")

    # 5) Fit Dixon–Coles (SOLO storico)
    run(f"{PY} step3b_profeta_fit_dc.py")

    # 6) Prediction FINALE (storico + fixture, con rho DC)
    run(f"{PY} step3_profeta_batch.py")

    # 7) Valutazione modello (SOLO storico)
    run(f"{PY} step4_profeta_eval.py")

    # =========================================================
    # FASE B — STATI DI PARTITA (DATA-DRIVEN)
    # =========================================================
    """
    Qui NON si fa prediction.
    Qui si costruisce CONOSCENZA STRUTTURALE
    partendo esclusivamente dallo storico.
    """

    # 8) CONTROL STATE
    #    (chi comanda: casa / ospite / equilibrio / draw-prone)
    run(f"{PY} step5a_profeta_control_state.py")

    # 9) GOAL STATE
    #    (ritmo partita: low / mid / high / wild goals)
    run(f"{PY} step5b_profeta_goal_state.py")

    # =========================================================
    # FASE C — PROFILI STATI × MERCATI (ANALISI OFFLINE)
    # =========================================================
    """
    Qui Profeta scopre:
    - quali mercati funzionano davvero
    - in quali combinazioni di stati
    - con che frequenze e supporti

    Serve per:
    - abilitare / disabilitare mercati
    - NON per generare alert diretti
    """

    # 10) CONTROL_STATE × GOAL_STATE → profilo mercati reali
    run(f"{PY} step6b_profeta_state_market_profile.py")

    print("\n✅ PIPELINE PROFETA COMPLETATA CON SUCCESSO")
    print("📌 Modello aggiornato + stati + profili pronti per runtime")


if __name__ == "__main__":
    main()