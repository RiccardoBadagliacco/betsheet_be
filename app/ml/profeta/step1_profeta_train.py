# ============================================================
# app/ml/profeta_v2/step1_profeta_train.py
# ============================================================

"""
STEP1 — Training del modello PROFETA V2 (Poisson + Feature Form)

Aggiunge:
    - beta_home (pesi feature squadra casa)
    - beta_away (pesi feature squadra ospite)

Formula:
    log λ_home = structural_home + beta_home ⋅ X_home
    log λ_away = structural_away + beta_away ⋅ X_away
"""

import sys
from pathlib import Path
import json

import pandas as pd
from tqdm import tqdm

import torch
from torch import nn


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR / "data"
STEP0_PATH = DATA_DIR / "step0_profeta.parquet"

OUT_PARAMS = DATA_DIR / "step1_profeta_params.pth"
OUT_META   = DATA_DIR / "step1_profeta_metadata.json"


# ============================================================
# MODEL
# ============================================================
class ProfetaModel(nn.Module):
    def __init__(self, n_leagues, n_seasons, n_team_seasons, n_features):
        super().__init__()

        # -------------------------
        # STRUTTURA BASE
        # -------------------------
        self.mu = nn.Parameter(torch.tensor(0.0))

        self.gamma_league = nn.Parameter(torch.zeros(n_leagues))
        self.hfa_league   = nn.Parameter(torch.zeros(n_leagues))
        self.delta_season = nn.Parameter(torch.zeros(n_seasons))

        self.att  = nn.Parameter(torch.zeros(n_team_seasons))
        self.defn = nn.Parameter(torch.zeros(n_team_seasons))

        # -------------------------
        # FEATURE HOME / AWAY
        # -------------------------
        self.beta_home = nn.Parameter(torch.zeros(n_features))
        self.beta_away = nn.Parameter(torch.zeros(n_features))

        self.n_features = n_features

    def forward(
        self,
        league_idx,
        season_idx,
        home_ts_idx,
        away_ts_idx,
        X_home,
        X_away
    ):
        # -------------------------
        # COMPONENTE STRUTTURALE
        # -------------------------
        base_home = (
            self.mu +
            self.gamma_league[league_idx] +
            self.delta_season[season_idx] +
            self.hfa_league[league_idx] +
            self.att[home_ts_idx] -
            self.defn[away_ts_idx]
        )

        base_away = (
            self.mu +
            self.gamma_league[league_idx] +
            self.delta_season[season_idx] +
            self.att[away_ts_idx] -
            self.defn[home_ts_idx]
        )

        # -------------------------
        # FEATURE
        # -------------------------
        lin_home = (X_home * self.beta_home).sum(dim=1)
        lin_away = (X_away * self.beta_away).sum(dim=1)

        log_h = base_home + lin_home
        log_a = base_away + lin_away

        # 🔒 sicurezza numerica: evita exp overflow
        log_h = torch.clamp(log_h, -10.0, 10.0)
        log_a = torch.clamp(log_a, -10.0, 10.0)

        return torch.exp(log_h), torch.exp(log_a)


# ============================================================
# TRAINING
# ============================================================
def train_profeta():
    torch.manual_seed(42)

    print("📥 Caricamento dataset STEP0:", STEP0_PATH)
    df = pd.read_parquet(STEP0_PATH)

    # ----------------------------------------
    # FILTRI STORICI
    # ----------------------------------------
    hist = df[df["is_fixture"] == False].copy()
    hist = hist.dropna(subset=["home_goals", "away_goals"])

    print(f"🧮 Match storici utilizzati per training: {len(hist)}")

    # ----------------------------------------
    # FEATURE DI FORMA
    # ----------------------------------------
    feature_home = [
        "pts_last5_home",
        "gf_last5_home",
        "ga_last5_home",
        "gd_last5_home",
    ]

    feature_away = [
        "pts_last5_away",
        "gf_last5_away",
        "ga_last5_away",
        "gd_last5_away",
    ]

    for col in feature_home + feature_away:
        if col not in hist.columns:
            raise ValueError(f"❌ Manca la colonna feature: {col}")

    # ----------------------------------------
    # FIX NaN/Inf nelle feature (inizio stagione)
    # ----------------------------------------
    feat_cols = feature_home + feature_away
    n_nan_before = int(hist[feat_cols].isna().sum().sum())
    if n_nan_before > 0:
        print(f"⚠️  NaN feature trovati: {n_nan_before} → fillna(0.0)")

    hist[feat_cols] = (
        hist[feat_cols]
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
    )

    # weight safety
    if "weight" not in hist.columns:
        hist["weight"] = 1.0
    hist["weight"] = (
        hist["weight"]
        .replace([float("inf"), float("-inf")], 1.0)
        .fillna(1.0)
        .clip(lower=1e-6)
    )

    # goals safety
    hist["home_goals"] = hist["home_goals"].fillna(0.0)
    hist["away_goals"] = hist["away_goals"].fillna(0.0)

    # ----------------------------------------
    # TENSOR INPUT
    # ----------------------------------------
    X_home = torch.tensor(hist[feature_home].values, dtype=torch.float32)
    X_away = torch.tensor(hist[feature_away].values, dtype=torch.float32)
    n_features = len(feature_home)

    # Debug rapido: controlli torch
    if not torch.isfinite(X_home).all() or not torch.isfinite(X_away).all():
        raise ValueError("❌ X_home/X_away contiene NaN o Inf anche dopo il fillna().")

    # ----------------------------------------
    # MAPPING INDICI
    # ----------------------------------------
    league_list = sorted(hist["league_id"].unique())
    season_list = sorted(hist["season_id"].unique())
    ts_list = sorted(
        pd.concat([hist["home_team_season_id"], hist["away_team_season_id"]]).unique()
    )

    league_to_idx = {lid: i for i, lid in enumerate(league_list)}
    season_to_idx = {sid: i for i, sid in enumerate(season_list)}
    ts_to_idx     = {ts: i for i, ts in enumerate(ts_list)}

    print(f"  • Leagues:       {len(league_list)}")
    print(f"  • Seasons:       {len(season_list)}")
    print(f"  • Team-seasons:  {len(ts_list)}")
    print(f"  • Features:      {n_features}")

    league_idx = torch.tensor(hist["league_id"].map(league_to_idx).values, dtype=torch.long)
    season_idx = torch.tensor(hist["season_id"].map(season_to_idx).values, dtype=torch.long)
    home_ts_idx = torch.tensor(hist["home_team_season_id"].map(ts_to_idx).values, dtype=torch.long)
    away_ts_idx = torch.tensor(hist["away_team_season_id"].map(ts_to_idx).values, dtype=torch.long)

    goals_h = torch.tensor(hist["home_goals"].values, dtype=torch.float32)
    goals_a = torch.tensor(hist["away_goals"].values, dtype=torch.float32)
    weights = torch.tensor(hist["weight"].values, dtype=torch.float32)

    if not torch.isfinite(goals_h).all() or not torch.isfinite(goals_a).all() or not torch.isfinite(weights).all():
        raise ValueError("❌ goals/weights contiene NaN o Inf.")

    # ----------------------------------------
    # MODELLO
    # ----------------------------------------
    model = ProfetaModel(
        n_leagues=len(league_list),
        n_seasons=len(season_list),
        n_team_seasons=len(ts_list),
        n_features=n_features,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    REG = 1e-5
    REG_FEATURE = 5e-3
    EPOCHS = 350

    print("🚀 Inizio training PROFETA V2 (+ Feature Form)…")

    for epoch in tqdm(range(EPOCHS), desc="Training progress"):
        optimizer.zero_grad()

        lam_h, lam_a = model(
            league_idx,
            season_idx,
            home_ts_idx,
            away_ts_idx,
            X_home,
            X_away
        )

        # Safety: lam deve essere > 0 e finita
        if not torch.isfinite(lam_h).all() or not torch.isfinite(lam_a).all():
            raise ValueError("❌ lambda contiene NaN/Inf (probabile overflow).")

        lam_h = torch.clamp(lam_h, min=1e-8, max=1e4)
        lam_a = torch.clamp(lam_a, min=1e-8, max=1e4)

        # Poisson NLL (senza costanti)
        nll = (
            lam_h - goals_h * torch.log(lam_h) +
            lam_a - goals_a * torch.log(lam_a)
        )
        loss_pois = (weights * nll).mean()

        # Regularizzazione classica
        reg_struct = (
            model.att.pow(2).mean() +
            model.defn.pow(2).mean() +
            model.gamma_league.pow(2).mean() +
            model.delta_season.pow(2).mean() +
            model.hfa_league.pow(2).mean()
        )

        # Regularizzazione forte sulle feature
        reg_feat = (
            model.beta_home.pow(2).mean() +
            model.beta_away.pow(2).mean()
        )

        loss = loss_pois + REG * reg_struct + REG_FEATURE * reg_feat

        if not torch.isfinite(loss):
            raise ValueError("❌ Loss è NaN/Inf. Controlla i dati (o riduci lr).")

        loss.backward()

        # (opzionale ma utile) evita esplosioni gradienti
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        if epoch % 25 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Loss={loss.item():.5f}")

    print("🎉 Training completato con successo!")

    torch.save(model.state_dict(), OUT_PARAMS)
    print("💾 Salvati parametri:", OUT_PARAMS)

    metadata = {
        "league_to_idx": league_to_idx,
        "season_to_idx": season_to_idx,
        "teamseason_to_idx": ts_to_idx,
        "feature_home": feature_home,
        "feature_away": feature_away,
        "n_features": n_features,
        "notes": {
            "nan_handling": "features fillna(0.0); weight clip>=1e-6",
            "numerical_safety": "log clamp [-10,10]; lambda clamp [1e-8,1e4]; grad clip 5.0"
        }
    }

    with open(OUT_META, "w") as f:
        json.dump(metadata, f, indent=2)

    print("💾 Salvato metadata:", OUT_META)
    print("✅ Training PROFETA V2 (+ Feature) COMPLETATO.")


if __name__ == "__main__":
    train_profeta()