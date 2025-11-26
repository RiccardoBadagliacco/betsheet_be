# app/ml/correlazioni_affini_v2/v2/high_confidence_search.py

import sys
from pathlib import Path
from typing import List, Dict, Any, Literal

import numpy as np
import pandas as pd

SignType = Literal["1", "2"]

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ------------------------------------------------------------
# Helper path (stile già usato negli altri script v2)
# ------------------------------------------------------------
def _resolve(names: List[str]) -> Path:
    script_dir = Path(__file__).resolve().parent
    local_data_dir = script_dir / "data"
    common_data_dir = script_dir.parent / "data"
    root_data_dir = REPO_ROOT / "data"

    for name in names:
        for base in (local_data_dir, common_data_dir, root_data_dir):
            candidate = base / name
            if candidate.exists():
                print(f"   ✔️  {name} trovato in: {candidate}")
                return candidate

    raise FileNotFoundError(
        f"File {names} non trovato. Cercato in {local_data_dir}, {common_data_dir}, {root_data_dir}"
    )


# ------------------------------------------------------------
# Caricamento datamaster + feature ausiliarie
# ------------------------------------------------------------
def load_datamaster() -> pd.DataFrame:
    print("📥 Carico datamaster...")
    dm_path = _resolve(["datamaster_1x2.parquet"])
    df = pd.read_parquet(dm_path)

    required_cols = [
        "pic_p1", "pic_px", "pic_p2",
        "bk_p1", "bk_px", "bk_p2",
        "avg_home_odds", "avg_away_odds",
        "is_home_win", "is_away_win",
        "fav_mod", "fav_book",
        "gap_mod",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(f"Datamaster manca colonna obbligatoria: {c}")

    # Entropia: se manca, riempi con NaN
    if "entropy_bk_1x2" not in df.columns:
        df["entropy_bk_1x2"] = np.nan

    # Probabilità favorita modello
    df["model_fav_prob"] = df[["pic_p1", "pic_px", "pic_p2"]].max(axis=1)
    # Probabilità favorita mercato
    df["market_fav_prob"] = df[["bk_p1", "bk_px", "bk_p2"]].max(axis=1)

    # Delta (modello - market) sulla favorita
    # Per farlo bene, scegliamo la col giusta per la favorita modello e mercato
    def _fav_prob(row, pref: str) -> float:
        if pref == "pic_p1":
            return row["pic_p1"]
        if pref == "pic_px":
            return row["pic_px"]
        if pref == "pic_p2":
            return row["pic_p2"]
        if pref == "bk_p1":
            return row["bk_p1"]
        if pref == "bk_px":
            return row["bk_px"]
        if pref == "bk_p2":
            return row["bk_p2"]
        return np.nan

    df["model_fav_prob_exact"] = df.apply(lambda r: _fav_prob(r, r["fav_mod"]), axis=1)
    df["market_fav_prob_exact"] = df.apply(lambda r: _fav_prob(r, r["fav_book"]), axis=1)
    df["delta_model_market_fav_prob"] = df["model_fav_prob_exact"] - df["market_fav_prob_exact"]

    # Coerenza favorita (già c'è mismatch_mod_book, ma ricalcoliamo un flag esplicito)
    df["fav_agreement"] = (df["fav_mod"] == "pic_p1") & (df["fav_book"] == "bk_p1") | \
                          (df["fav_mod"] == "pic_p2") & (df["fav_book"] == "bk_p2")

    return df


# ------------------------------------------------------------
# Funzione di valutazione filtri
# ------------------------------------------------------------
def evaluate_filters_for_sign(
    df: pd.DataFrame,
    sign: SignType,
    min_matches: int = 300,
    min_acc: float = 0.70,
) -> pd.DataFrame:
    """
    Cerca combinazioni di filtri che massimizzano l'accuracy sul segno 1 o 2.
    """

    print(f"\n🔍 Cerco filtri high-confidence per il segno {sign}...")

    # Definizione base: partite dove SIA il modello che il mercato hanno come favorita 1 o 2
    if sign == "1":
        base_mask = (df["fav_mod"] == "pic_p1") & (df["fav_book"] == "bk_p1")
        fav_odds_col = "avg_home_odds"
        outcome_col = "is_home_win"
    else:  # "2"
        base_mask = (df["fav_mod"] == "pic_p2") & (df["fav_book"] == "bk_p2")
        fav_odds_col = "avg_away_odds"
        outcome_col = "is_away_win"

    df_base = df[base_mask].copy()
    if df_base.empty:
        print(f"⚠️ Nessun match con favorita coerente per segno {sign}.")
        return pd.DataFrame()

    # Griglie di parametri (puoi modificarle per essere più/meno aggressive)
    model_prob_thresholds = [0.60, 0.65, 0.70, 0.75]
    market_prob_thresholds = [0.55, 0.60, 0.65, 0.70]
    gap_thresholds = [0.15, 0.20, 0.25, 0.30]
    fav_odds_max_list = [1.30, 1.40, 1.50, 1.60, 1.70, 1.80]
    entropy_max_list = [1.25, 1.20, 1.15, 1.10]  # sempre più restrittivo

    results: list[Dict[str, Any]] = []
    total_combos = (
        len(model_prob_thresholds)
        * len(market_prob_thresholds)
        * len(gap_thresholds)
        * len(fav_odds_max_list)
        * len(entropy_max_list)
    )
    combo_idx = 0

    for m_th in model_prob_thresholds:
        for mk_th in market_prob_thresholds:
            for g_th in gap_thresholds:
                for fav_odds_max in fav_odds_max_list:
                    for ent_max in entropy_max_list:
                        combo_idx += 1
                        # Costruisco maschera
                        mask = (
                            (df_base["model_fav_prob"] >= m_th)
                            & (df_base["market_fav_prob"] >= mk_th)
                            & (df_base["gap_mod"] >= g_th)
                            & (df_base[fav_odds_col] <= fav_odds_max)
                        )

                        # entropia: se NaN, non vogliamo escludere per quello, quindi:
                        ent = df_base["entropy_bk_1x2"]
                        mask = mask & ((ent <= ent_max) | (ent.isna()))

                        subset = df_base[mask]
                        n = len(subset)
                        if n < min_matches:
                            continue

                        # Accuracy sul segno principale
                        acc = subset[outcome_col].mean()

                        if acc < min_acc:
                            continue

                        # Alcune statistiche aggiuntive
                        p1 = subset["is_home_win"].mean()
                        pX = subset["is_draw"].mean() if "is_draw" in subset.columns else np.nan
                        p2 = subset["is_away_win"].mean()

                        results.append(
                            {
                                "sign": sign,
                                "n_matches": n,
                                "accuracy_main": float(acc),
                                "p1": float(p1),
                                "pX": float(pX) if pX == pX else np.nan,
                                "p2": float(p2),
                                "model_prob_min": m_th,
                                "market_prob_min": mk_th,
                                "gap_mod_min": g_th,
                                "fav_odds_max": fav_odds_max,
                                "entropy_max": ent_max,
                                "avg_model_fav_prob": subset["model_fav_prob"].mean(),
                                "avg_market_fav_prob": subset["market_fav_prob"].mean(),
                                "avg_delta_model_market": subset["delta_model_market_fav_prob"].mean(),
                                "avg_fav_odds": subset[fav_odds_col].mean(),
                            }
                        )

    if not results:
        print(f"⚠️ Nessuna combinazione soddisfa min_matches={min_matches} e min_acc={min_acc:.2f} per segno {sign}.")
        return pd.DataFrame()

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(
        ["accuracy_main", "n_matches"], ascending=[False, False]
    ).reset_index(drop=True)
    return res_df


def main():
    df = load_datamaster()

    # Parametri globali (puoi modificarli)
    min_matches = 300
    min_acc = 0.70

    # Segno 1
    res1 = evaluate_filters_for_sign(df, "1", min_matches=min_matches, min_acc=min_acc)
    # Segno 2
    res2 = evaluate_filters_for_sign(df, "2", min_matches=min_matches, min_acc=min_acc)

    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not res1.empty:
        out1 = out_dir / "high_conf_filters_sign1.csv"
        res1.to_csv(out1, index=False)
        print(f"\n💾 Salvato filtri high-confidence segno 1 in {out1}")
        print("\n🏆 TOP 10 filtri segno 1:")
        print(res1.head(10).to_string(index=False))

    if not res2.empty:
        out2 = out_dir / "high_conf_filters_sign2.csv"
        res2.to_csv(out2, index=False)
        print(f"\n💾 Salvato filtri high-confidence segno 2 in {out2}")
        print("\n🏆 TOP 10 filtri segno 2:")
        print(res2.head(10).to_string(index=False))

    if res1.empty and res2.empty:
        print("\n❌ Nessun filtro high-confidence trovato con le soglie attuali.")
        print("   Prova ad abbassare min_acc o min_matches dentro lo script.")


if __name__ == "__main__":
    main()