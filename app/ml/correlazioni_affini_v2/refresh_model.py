# refresh_model.py
import subprocess
import time
import sys
import os

# ------------------------------------------------------
# 1. STEPS COMPLETI (TRAIN = RIADDESTRAMENTO)
# ------------------------------------------------------
STEPS_TRAIN = [
    "step0/step0_dataset_base_features.py",
    "step1/step1a_elo_v2.py",
    "step1/step1b_form_recent.py",
    "step1/step1z_compute_fixture_features.py",
    "step1/step1c_build_dataset_with_elo_form.py",
    "step2/step2a_picchetto_tech_v2.py",
    "step2/step2b_ou15_features.py",
    "step2/step2b_ou25_features.py",
    "step2/step2b_1X2_features.py",
    "step3/step3a_1x2_clustering_v2.py",
    "step3/step3b_ou25_clustering_v2.py",
    "step3/step3c_ou15_clustering_v2.py",
    "step4/step4a_affini_index_wide_v2.py",
    "step4/step4b_affini_index_slim_v2.py",


    # 🟦 NUOVI STEP NECESSARI
    "meta/stepX1_build_master_1x2.py",
    "meta/stepX2_train_meta_model_1x2.py",
    "meta/stepX3_meta_catboost_1x2_v3.py",
    "meta/stepX4_calibrate_meta_1x2.py",
    "meta/stepX6_meta_backtest_1x2.py",
    "meta/stepX7_meta_backtest_ev_1x2.py",
    "meta/stepX5_build_bet_global_stats.py",


]

# ------------------------------------------------------
# 2. STEPS FULL (NO TRAINING)
# ------------------------------------------------------
STEPS_FULL = [
    "step0/step0_dataset_base_features.py",
    "step1/step1a_elo_v2.py",
    "step1/step1b_form_recent.py",
    "step1/step1z_compute_fixture_features.py",
    "step1/step1c_build_dataset_with_elo_form.py",
    "step2/step2a_picchetto_tech_v2.py",
    "step2/step2b_ou15_features.py",
    "step2/step2b_ou25_features.py",
    "step2/step2b_1X2_features.py",
    "step4/step4a_affini_index_wide_v2.py",
    "step4/step4b_affini_index_slim_v2.py",

    # 🟦 NUOVI STEP X
    "meta/stepX1_build_master_1x2.py",
    "meta/stepX2_train_meta_model_1x2.py",
    "meta/stepX3_meta_catboost_1x2_v3.py",
    "meta/stepX4_calibrate_meta_1x2.py",
    "meta/stepX5_build_bet_global_stats.py",
    "meta/stepX6_meta_backtest_1x2.py",
    "meta/stepX7_meta_backtest_ev_1x2.py"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------
# 3. RUN STEP
# ------------------------------------------------------
def run_step(script_name, env=None):
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
        stderr=sys.stderr,
        env=env or os.environ.copy(),
    )
    end = time.time()

    if proc.returncode != 0:
        print(f"\n❌ Step FALLITO: {script_name}")
        sys.exit(proc.returncode)

    print(f"\n✅ Step completato in {end - start:.2f} sec")


# ------------------------------------------------------
# 4. MAIN
# ------------------------------------------------------
def main():
    mode = "train"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    print("\n==============================================")
    print(f"  🔄 MODEL REFRESH PIPELINE – MODE: {mode.upper()}")
    print("==============================================\n")

    # ----------------------------
    # SELECT STEPS
    # ----------------------------
    if mode == "full":
        steps = STEPS_FULL
        include_fixtures = "1"
    else:
        steps = STEPS_TRAIN
        include_fixtures = "0"

    # ----------------------------
    # ENV VAR per step0
    # ----------------------------
    env = os.environ.copy()
    env["INCLUDE_FIXTURES"] = include_fixtures

    for step in steps:
        run_step(step, env)

    print("\n==============================================")
    print("  🎉 MODEL REFRESH COMPLETATO SENZA ERRORI")
    print("==============================================\n")


if __name__ == "__main__":
    main()