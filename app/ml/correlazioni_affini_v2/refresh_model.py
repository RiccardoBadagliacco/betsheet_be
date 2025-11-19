# refresh_model.py
import subprocess
import time
import sys
import os

STEPS = [
    "step0/step0_dataset_base_features.py",
    "step1/step1a_elo_v2.py",
    "step1/step1c_build_dataset_with_elo_form.py",
    "step2/step2b_ou15_features.py",
    "step2/step2b_ou25_features.py",
    "step2/step2b_1X2_features.py",
    "step3/step3a_1x2_clustering_v2.py",
    "step3/step3b_ou25_clustering_v2.py",
    "step3/step3c_ou15_clustering_v2.py",
    "step4/step4a_affini_index_wide_v2.py",
    "step4/step4b_affini_index_slim_v2.py",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_step(script_name):
    print("\n" + "="*70)
    print(f"▶ Running step: {script_name}")
    print("="*70)

    path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(path):
        print(f"❌ ERRORE: Script non trovato: {path}")
        sys.exit(1)

    start = time.time()
    proc = subprocess.run(
        ["python", path],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    end = time.time()

    if proc.returncode != 0:
        print(f"\n❌ Step FALLITO: {script_name}")
        sys.exit(proc.returncode)

    print(f"\n✅ Step completato in {end - start:.2f} sec")


def main():
    print("\n==============================================")
    print("  🔄 MODEL REFRESH PIPELINE – FULL REBUILD")
    print("==============================================\n")

    for step in STEPS:
        run_step(step)

    print("\n==============================================")
    print("  🎉 MODEL REFRESH COMPLETATO SENZA ERRORI")
    print("==============================================\n")

if __name__ == "__main__":
    main()