# app/ml/correlazioni_affini/cluster_engine.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib


class ClusterEngine:
    """
    Motore centrale per:
      - caricamento df con assegnazioni cluster (step3_cluster_assignments.parquet)
      - caricamento scaler / PCA / GMM
      - calcolo profilo statistico per ogni cluster
      - prediction cluster + profilo da:
          * match_id presente nel parquet
          * dizionario di feature (es. odds, ecc.)
    """

    def __init__(self) -> None:
        base = Path(__file__).resolve().parents[1]  # .../app/ml
        data_dir = base / "correlazioni_affini" / "data"
        models_dir = base / "correlazioni_affini" / "models"

        self.data_dir = data_dir
        self.models_dir = models_dir

        # ---------------------------------------------------------
        # 1) Carico dataset con assegnazioni cluster
        # ---------------------------------------------------------
        assign_path = data_dir / "step3_cluster_assignments.parquet"
        if not assign_path.exists():
            raise FileNotFoundError(f"File assegnazioni cluster non trovato: {assign_path}")

        df = pd.read_parquet(assign_path)
        self.df_assign = df

        # ---------------------------------------------------------
        # 2) Ricostruisco le feature numeriche COME in step3_train_gmm_clusters
        # ---------------------------------------------------------
        # In training, feature_cols = [
        #   c for c in df.columns
        #   if c not in ["match_id", "league", "country", "season",
        #                "home_team", "away_team"]
        #   and df[c].dtype != "object"
        # ]
        #
        # Qui in step3_cluster_assignments è già presente "cluster" (int),
        # che NON era usato in training, quindi va escluso esplicitamente.
        excluded = {"match_id", "league", "country", "season",
                    "home_team", "away_team", "cluster"}

        feature_cols: List[str] = [
            c
            for c in df.columns
            if c not in excluded and df[c].dtype != "object"
        ]
        self.feature_cols = feature_cols

        # Pre-calcolo le medie di colonna per riempire i NaN / valori mancanti
        self.col_means = df[feature_cols].mean()

        # ---------------------------------------------------------
        # 3) Carico scaler / PCA / GMM
        # ---------------------------------------------------------
        scaler_path = models_dir / "scaler_full.pkl"
        pca_path = models_dir / "pca_full.pkl"
        gmm_path = models_dir / "gmm_model_full.pkl"

        if not scaler_path.exists() or not pca_path.exists() or not gmm_path.exists():
            raise FileNotFoundError(
                f"Modelli mancanti in {models_dir}. "
                f"Assicurati di aver eseguito step3_train_gmm_clusters.py."
            )

        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        self.gmm = joblib.load(gmm_path)

        # Safety check: numero feature coerente
        if len(self.feature_cols) != self.scaler.n_features_in_:
            raise ValueError(
                f"Incoerenza feature: scaler attende {self.scaler.n_features_in_} "
                f"ma feature_cols ha len={len(self.feature_cols)}. "
                f"feature_cols = {self.feature_cols}"
            )

        # ---------------------------------------------------------
        # 4) Costruisco i profili per ciascun cluster
        # ---------------------------------------------------------
        self.cluster_profiles: Dict[int, Dict[str, Any]] = self._build_cluster_profiles()

    # ======================================================================
    #  UTILS PROFILING
    # ======================================================================

    @staticmethod
    def _pct(mask: pd.Series | np.ndarray) -> float:
        """Percentuale [0–100] di True in una serie booleana."""
        arr = mask.to_numpy() if isinstance(mask, pd.Series) else np.asarray(mask)
        if arr.size == 0:
            return 0.0
        return float(100.0 * arr.mean())

    def _profile_subcluster(self, sub: pd.DataFrame) -> Dict[str, Any]:
        """
        Crea il profilo statistico di un singolo cluster (sotto-dataframe).
        Usa SOLO informazioni di risultato (home_ft, away_ft).
        """
        n = len(sub)
        if n == 0:
            return {
                "n_matches": 0,
                "1X2": {"1": 0.0, "X": 0.0, "2": 0.0},
                "OU": {"O15": 0.0, "O25": 0.0, "O35": 0.0},
                "GG": {"gg": 0.0, "ng": 0.0},
                "MG_total": {"1_3": 0.0, "1_4": 0.0, "1_5": 0.0},
                "MG_home": {"1_3": 0.0, "1_4": 0.0, "1_5": 0.0},
                "MG_away": {"1_3": 0.0, "1_4": 0.0, "1_5": 0.0},
                "top_scores": [],
            }

        tg = sub["home_ft"] + sub["away_ft"]
        home_g = sub["home_ft"]
        away_g = sub["away_ft"]

        # 1X2
        res_1 = self._pct(sub["result_1x2"] == "1") / 100.0
        res_x = self._pct(sub["result_1x2"] == "X") / 100.0
        res_2 = self._pct(sub["result_1x2"] == "2") / 100.0

        # Over/Under
        ou15 = self._pct(tg >= 2) / 100.0
        ou25 = self._pct(tg >= 3) / 100.0
        ou35 = self._pct(tg >= 4) / 100.0

        # GG / NG
        gg = self._pct((home_g > 0) & (away_g > 0)) / 100.0
        ng = 1.0 - gg

        # MG totali
        mg_total_1_3 = self._pct((tg >= 1) & (tg <= 3)) / 100.0
        mg_total_1_4 = self._pct((tg >= 1) & (tg <= 4)) / 100.0
        mg_total_1_5 = self._pct((tg >= 1) & (tg <= 5)) / 100.0

        # MG casa
        mg_home_1_3 = self._pct((home_g >= 1) & (home_g <= 3)) / 100.0
        mg_home_1_4 = self._pct((home_g >= 1) & (home_g <= 4)) / 100.0
        mg_home_1_5 = self._pct((home_g >= 1) & (home_g <= 5)) / 100.0

        # MG ospite
        mg_away_1_3 = self._pct((away_g >= 1) & (away_g <= 3)) / 100.0
        mg_away_1_4 = self._pct((away_g >= 1) & (away_g <= 4)) / 100.0
        mg_away_1_5 = self._pct((away_g >= 1) & (away_g <= 5)) / 100.0

        # Scoreline più frequenti
        scores = home_g.astype(str) + "-" + away_g.astype(str)
        vc = scores.value_counts(normalize=True).head(3)
        top_scores = [{"score": s, "freq": float(freq)} for s, freq in vc.items()]

        return {
            "n_matches": int(n),
            "1X2": {"1": res_1, "X": res_x, "2": res_2},
            "OU": {"O15": ou15, "O25": ou25, "O35": ou35},
            "GG": {"gg": gg, "ng": ng},
            "MG_total": {"1_3": mg_total_1_3, "1_4": mg_total_1_4, "1_5": mg_total_1_5},
            "MG_home": {"1_3": mg_home_1_3, "1_4": mg_home_1_4, "1_5": mg_home_1_5},
            "MG_away": {"1_3": mg_away_1_3, "1_4": mg_away_1_4, "1_5": mg_away_1_5},
            "top_scores": top_scores,
        }

    def _build_cluster_profiles(self) -> Dict[int, Dict[str, Any]]:
        """
        Costruisce il profilo per ciascun cluster a partire da df_assign.
        """
        df = self.df_assign.copy()

        # colonne di risultato già presenti: home_ft, away_ft
        df["total_goals"] = df["home_ft"] + df["away_ft"]
        df["result_1x2"] = np.where(
            df["home_ft"] > df["away_ft"],
            "1",
            np.where(df["home_ft"] < df["away_ft"], "2", "X"),
        )

        profiles: Dict[int, Dict[str, Any]] = {}
        for cid, sub in df.groupby("cluster"):
            profiles[int(cid)] = self._profile_subcluster(sub)

        return profiles

    # ======================================================================
    #  FEATURE VECTOR + PREDICTION
    # ======================================================================

    def build_feature_vector(self, row_dict: Dict[str, Any]) -> np.ndarray:
        """
        Costruisce un vettore di feature (np.ndarray shape=(n_features,))
        nel **medesimo ordine** di self.feature_cols.

        Logica:
          - si parte dalle medie di colonna (self.col_means)
          - poi si sovrascrivono i valori presenti in row_dict
          - eventuali None / NaN vengono ignorati (resta la media)
        """
        x = self.col_means.copy()  # Series indicizzata per feature_cols

        for col in self.feature_cols:
            if col in row_dict:
                val = row_dict[col]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    x[col] = val

        # ritorno come array nella giusta order
        return x[self.feature_cols].astype(float).to_numpy()

    def predict_cluster(self, row_dict: Dict[str, Any]) -> int:
        """
        Predice il cluster per una riga di feature parziale.
        """
        x = self.build_feature_vector(row_dict)
        xs = self.scaler.transform([x])
        xp = self.pca.transform(xs)
        cid = int(self.gmm.predict(xp)[0])
        return cid

    def profile_cluster(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """
        Restituisce il profilo del cluster (o None se non esiste).
        """
        return self.cluster_profiles.get(int(cluster_id))

    def predict_cluster_with_profile(
        self, row_dict: Dict[str, Any]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Predice cluster e attach il profilo.
        """
        cid = self.predict_cluster(row_dict)
        return cid, self.profile_cluster(cid)

    # ======================================================================
    #  UTILITIES PER MATCH_ID STORICI
    # ======================================================================

    @property
    def cluster_ids(self) -> List[int]:
        return sorted(int(c) for c in self.cluster_profiles.keys())

    def profile_match_id(self, match_id: str) -> Optional[Tuple[int, Dict[str, Any]]]:
        """
        Ritorna (cluster_id, profilo_cluster) per un match_id presente nel parquet.
        """
        sub = self.df_assign[self.df_assign["match_id"] == match_id]
        if sub.empty:
            return None
        cid = int(sub["cluster"].iloc[0])
        return cid, self.profile_cluster(cid)


# Singleton globale, utilizzabile in tutto il progetto
CLUSTER_ENGINE = ClusterEngine()