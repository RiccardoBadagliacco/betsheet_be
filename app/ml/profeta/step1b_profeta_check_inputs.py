from pathlib import Path
import pandas as pd
import numpy as np

FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR / "data"
STEP0_PATH = DATA_DIR / "step0_profeta.parquet"

feature_home = ["pts_last5_home", "gf_last5_home", "ga_last5_home", "gd_last5_home"]
feature_away = ["pts_last5_away", "gf_last5_away", "ga_last5_away", "gd_last5_away"]

df = pd.read_parquet(STEP0_PATH)
hist = df[df["is_fixture"] == False].copy()

cols = ["home_goals","away_goals","weight"] + feature_home + feature_away

print("rows hist:", len(hist))
print("missing columns:", [c for c in cols if c not in hist.columns])

# conta NaN
nan_counts = hist[cols].isna().sum().sort_values(ascending=False)
print("\nNaN counts (top):")
print(nan_counts.head(20))

# conta inf
inf_counts = pd.Series({
    c: np.isinf(hist[c].astype(float)).sum() if c in hist.columns else None
    for c in cols
}).sort_values(ascending=False)
print("\nInf counts:")
print(inf_counts)

# range sanity
print("\nweight describe:")
print(hist["weight"].describe())

print("\ngoals describe:")
print(hist[["home_goals","away_goals"]].describe())