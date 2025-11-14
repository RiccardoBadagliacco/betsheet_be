# app/ml/correlazioni_affini/refresh_model.py

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

# Ordine completo della pipeline
STEPS = [
    # STEP 0 – dataset base + features base
    ("STEP 0 → dataset base + features", BASE / "step0" / "step0_dataset_base_features.py"),

    # STEP 1 – Elo, Poisson, form, dataset con elo
    ("STEP 1a → Elo ratings",          BASE / "step1" / "step1a_elo.py"),
    ("STEP 1b → Poisson expected goals", BASE / "step1" / "step1b_poisson.py"),
    ("STEP 1c → Form regression",      BASE / "step1" / "step1c_form_regression_pro.py"),
    ("STEP 1d → Dataset con Elo",      BASE / "step1" / "step1d_build_dataset_base_with_elo.py"),

    # STEP 2 – Picchetto + feature per clustering
    ("STEP 2a → Picchetto tecnico",    BASE / "step2" / "step2a_picchetto_tech_pro.py"),
    ("STEP 2b → Feature per GMM",      BASE / "step2" / "step2b_build_cluster_features.py"),

    # STEP 3 – Train GMM + assegnazione cluster
    ("STEP 3 → Train GMM + clusters",  BASE / "step3_train_gmm_clusters.py"),

    # STEP 4 – Profilazione cluster
    ("STEP 4 → Profilazione cluster",  BASE / "step4_profile_gmm_clusters.py"),
]


def run_step(name: str, script_path: Path) -> None:
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔵 {name}")
    print(f"📄 Script: {script_path}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if not script_path.exists():
        print(f"❌ ERRORE: file non trovato: {script_path}")
        sys.exit(1)

    # Esegue lo script Python e cattura output
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print("❌ ERRORE durante l'esecuzione dello step!")
        if result.stderr:
            print("── STDERR ─────────────────────────")
            print(result.stderr)
        sys.exit(result.returncode)

    print("✅ Step completato con successo")


def main():
    print("\n===============================================")
    print("   🔄 REFRESH COMPLETO MODELLO CORRELAZIONI AFFINI (GMM)")
    print("===============================================\n")

    for name, script in STEPS:
        run_step(name, script)

    print("\n===============================================")
    print("   🎉 PIPELINE COMPLETATA! Modello GMM aggiornato.")
    print("===============================================\n")


if __name__ == "__main__":
    main()