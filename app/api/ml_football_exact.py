#!/usr/bin/env python3
"""
ml_football_exact.py — Refactor 2025
===================================

Obiettivi del refactor:
- Eliminare duplicazioni (predict_match definita due volte)
- Separare helpers riutilizzabili (fixture row, cleaning, fallback odds)
- Introdurre caching leggero per predictor/dataset per lega
- Mantenere compatibilità API ed integrazione con betting_recommendations
- Migliorare robustezza (type hints, error handling, input sanitization)

Dipendenze interne:
- addons.football_utils: get_team_features, remove_vig, estimate_lambdas_from_market,
  estimate_lambdas_from_stats, calculate_probabilities
- addons.betting_recommendations: get_recommended_bets
- addons.context_scoring_v4: opzionale (usato dai servizi esterni / backtest)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from addons.betting_config import MARKET_KEY_MAP


# --- addons imports (funzioni condivise) ---
from addons.football_utils import (
    get_team_features,
    remove_vig,
    estimate_lambdas_from_market,
    estimate_lambdas_from_stats,
    calculate_probabilities,
)
from addons.betting_recommendations import get_recommended_bets
from addons.compute_team_profiles import annotate_pre_match_elo,build_team_profiles

# ==============================================================
# Router FastAPI
# ==============================================================
router = APIRouter()

# ==============================================================
# Helpers e utilità locali
# ==============================================================

def _convert_numpy_types(obj: Any) -> Any:
    """Converte tipi numpy in tipi Python nativi (JSON-safe)."""
    if obj is None:
        return None
    try:
        import numpy as _np
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, ( _np.integer, _np.int32, _np.int64 )):
            return int(obj)
        if isinstance(obj, ( _np.floating, _np.float32, _np.float64 )):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(v) for v in obj]
    return obj


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _fallback_odds_map() -> Dict[str, float]:
    """Quote di fallback conservative nel caso manchino dal DB."""
    return {
        'AvgH': 2.00,
        'AvgD': 3.20,
        'AvgA': 3.80,
        'Avg>2.5': 1.80,
        'Avg<2.5': 2.00,
    }


def _build_fixture_row(fixture) -> Dict[str, Any]:
    """Converte un oggetto Fixture in una riga compatibile col DataFrame storico.
    Applica fallback odds se mancanti.
    """
    fh = getattr(fixture, 'avg_home_odds', None)
    fd = getattr(fixture, 'avg_draw_odds', None)
    fa = getattr(fixture, 'avg_away_odds', None)
    fo = getattr(fixture, 'avg_over_25_odds', None)
    fu = getattr(fixture, 'avg_under_25_odds', None)
    fb = _fallback_odds_map()

    return {
        'Date': pd.to_datetime(getattr(fixture, 'match_date', None)),
        'HomeTeam': fixture.home_team.name if getattr(fixture, 'home_team', None) else 'Unknown',
        'AwayTeam': fixture.away_team.name if getattr(fixture, 'away_team', None) else 'Unknown',
        'FTHG': getattr(fixture, 'home_goals_ft', None),
        'FTAG': getattr(fixture, 'away_goals_ft', None),
        'AvgH': fh if fh is not None else fb['AvgH'],
        'AvgD': fd if fd is not None else fb['AvgD'],
        'AvgA': fa if fa is not None else fb['AvgA'],
        'Avg>2.5': fo if fo is not None else fb['Avg>2.5'],
        'Avg<2.5': fu if fu is not None else fb['Avg<2.5'],
    }


# ==============================================================
# Predictor (solo logica di previsione)
# ==============================================================

@dataclass
class PredictorConfig:
    global_window: int = 10
    venue_window: int = 5
    market_weight: float = 0.60


class ExactSimpleFooballPredictor:
    """Replica del SimpleFootballPredictor adattata a DataFrame DB.
    Mantiene la stessa logica di calcolo (features, lambdas, Poisson).
    """

    def __init__(self, config: PredictorConfig | None = None):
        self.cfg = config or PredictorConfig()
        self.data: Optional[pd.DataFrame] = None
        # ✅ Compatibilità + comportamento aggiornato
        self.use_context_scoring: bool = True
        self.model_version: str = "CONTEXT_SCORING_V4"

    # ----------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------
    def load_data(self, db: Session, league_code: str) -> pd.DataFrame:
        """Carica i match della lega dal DB e li normalizza."""
        from app.db.models_football import Match, Team, League, Season

        q = (
            db.query(Match)
              .join(Season)
              .join(League)
              .filter(League.code == league_code)
              .order_by(Match.match_date)
        )
        matches = q.all()
        if not matches:
            raise ValueError(f"No matches found for league {league_code}")

        rows: List[Dict[str, Any]] = []
        for m in matches:
            rows.append({
                'Date': m.match_date,
                'HomeTeam': m.home_team.name if m.home_team else 'Unknown',
                'AwayTeam': m.away_team.name if m.away_team else 'Unknown',
                'FTHG': m.home_goals_ft,
                'FTAG': m.away_goals_ft,
                'AvgH': m.avg_home_odds,
                'AvgD': m.avg_draw_odds,
                'AvgA': m.avg_away_odds,
                'Avg>2.5': m.avg_over_25_odds,
                'Avg<2.5': m.avg_under_25_odds,
            })
        df = pd.DataFrame(rows)

        # Normalizzazione
        df['Date'] = pd.to_datetime(df['Date'])
        df['HomeTeam'] = df['HomeTeam'].astype(str).str.strip()
        df['AwayTeam'] = df['AwayTeam'].astype(str).str.strip()
        for col in ['FTHG', 'FTAG', 'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
        
        df = annotate_pre_match_elo(df)
        
        self.team_profile = build_team_profiles(df)

        self.df_full = df
        return df

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------
    def predict_match(self, df: pd.DataFrame, match_idx: int) -> Dict[str, Any]:
        
        if not hasattr(self, "df_full") or self.df_full is None:
            self.df_full = df
            
        """Predice un singolo match (probabilità mercati + metadati)."""
        match = df.iloc[match_idx]
        current_date = match['Date']
        
        # Features squadra (rolling)
        home_features = get_team_features(df, match['HomeTeam'], current_date, is_home=True)
        away_features = get_team_features(df, match['AwayTeam'], current_date, is_home=False)

        if not home_features.get('valid', False) or not away_features.get('valid', False):
            print(f"[SKIP] Storico insufficiente per {match['HomeTeam']} ({home_features.get('total_matches', 0)}) "
                f"o {match['AwayTeam']} ({away_features.get('total_matches', 0)})")
            return None
        

        # Lambdas dal mercato (se quote presenti)
        market_lambdas: Tuple[float, float] = (1.3, 1.1)  # default nel caso manchino quote
        if pd.notna(match.get('AvgH')) and pd.notna(match.get('AvgD')) and pd.notna(match.get('AvgA')):
            odds_1x2 = {'H': match['AvgH'], 'D': match['AvgD'], 'A': match['AvgA']}
            p_1x2 = remove_vig(odds_1x2)
            odds_ou = {'over': match.get('Avg>2.5', 1.9), 'under': match.get('Avg<2.5', 1.9)}
            p_ou = remove_vig(odds_ou)
            market_lambdas = estimate_lambdas_from_market(p_1x2, p_ou)

        # Lambdas da statistiche
        
        current_date = match['Date']
        matchday_home = df[df['Date'] < current_date].groupby('HomeTeam').size().get(match['HomeTeam'], 0)
        matchday_away = df[df['Date'] < current_date].groupby('AwayTeam').size().get(match['AwayTeam'], 0)
        matchday = max(matchday_home, matchday_away)

        source_df = df

        # Se mancano le colonne ELO, calcolale ora sul DF corrente
        if 'elo_home_pre' not in source_df.columns or 'elo_away_pre' not in source_df.columns:
            source_df = annotate_pre_match_elo(source_df)

        # Leggi l'ultima riga con il match_idx sullo STESSO DF
        row_elo = source_df.iloc[match_idx]
        elo_home_raw = row_elo.get('elo_home_pre')
        elo_away_raw = row_elo.get('elo_away_pre')

        # 👇 fallback automatico se il valore è None o NaN
        elo_home_pre = float(elo_home_raw) if pd.notna(elo_home_raw) and elo_home_raw is not None else 1500.0
        elo_away_pre = float(elo_away_raw) if pd.notna(elo_away_raw) and elo_away_raw is not None else 1500.0

        lambda_home, lambda_away = estimate_lambdas_from_stats(
            home_features, away_features,
            match['HomeTeam'], match['AwayTeam'],
            matchday=matchday,
            elo_home_pre=elo_home_pre, elo_away_pre=elo_away_pre,
            team_profile=getattr(self, "team_profile", None)
        )

        matchday_home = df[df["Date"] < current_date].groupby("HomeTeam").size().get(match["HomeTeam"], 0)
        matchday_away = df[df["Date"] < current_date].groupby("AwayTeam").size().get(match["AwayTeam"], 0)
        matchday = max(matchday_home, matchday_away)
        stats_lambdas = estimate_lambdas_from_stats(home_features, away_features, match["HomeTeam"], match["AwayTeam"], matchday)

        # Probabilità mercati dalla matrice di Poisson
        probs = calculate_probabilities(lambda_home, lambda_away)
        
        # ============================
        # 🔁 Mappa chiavi interne → mercati leggibili
        # ============================
        probs = {MARKET_KEY_MAP.get(k, k): v for k, v in probs.items()}

        # Risultato finale
        result: Dict[str, Any] = {
            'match_idx': match_idx,
            'date': current_date,
            'home_team': match['HomeTeam'],
            'away_team': match['AwayTeam'],
            'lambda_home': float(lambda_home),
            'lambda_away': float(lambda_away),
            'lambda_home_market': float(market_lambdas[0]),
            'lambda_away_market': float(market_lambdas[1]),
            'lambda_home_stats': float(stats_lambdas[0]),
            'lambda_away_stats': float(stats_lambdas[1]),
            'home_matches_count': int(home_features.get('total_matches', 0)),
            'away_matches_count': int(away_features.get('total_matches', 0)),
            **probs,
        }

        # Quote 1X2 (se presenti)
        if pd.notna(match.get('AvgH')) and pd.notna(match.get('AvgD')) and pd.notna(match.get('AvgA')):
            result['odds_1'] = float(match['AvgH'])
            result['odds_X'] = float(match['AvgD'])
            result['odds_2'] = float(match['AvgA'])

        # Risultato reale (se storico)
        if pd.notna(match.get('FTHG')) and pd.notna(match.get('FTAG')):
            result['actual_home_goals'] = int(match['FTHG'])
            result['actual_away_goals'] = int(match['FTAG'])
            result['actual_total_goals'] = int(match['FTHG']) + int(match['FTAG'])
            result['actual_scoreline'] = f"{int(match['FTHG'])}-{int(match['FTAG'])}"

        return result

    def predict_matches(self, df: pd.DataFrame, start_idx: int = 50) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for idx in range(start_idx, len(df)):
            try:
                results.append(self.predict_match(df, idx))
            except Exception:
                # Non bloccare l'intero batch per un match
                continue
        return results

# ==============================================================
# Caching predictor per lega (riduce tempi di I/O)
# ==============================================================

_LEAGUE_PREDICTORS: Dict[str, ExactSimpleFooballPredictor] = {}


def _get_model_for_league(league_code: str) -> ExactSimpleFooballPredictor:
    pred = _LEAGUE_PREDICTORS.get(league_code)
    if pred is None:
        pred = ExactSimpleFooballPredictor()
        _LEAGUE_PREDICTORS[league_code] = pred
    return pred

# ==============================================================
# Connessione DB (read-only helper)
# ==============================================================

def get_football_db():
    """Crea una sessione SQLite locale per il dataset football (read-only scope)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    FOOTBALL_DATABASE_URL = "sqlite:///./data/football_dataset.db"
    engine = create_engine(FOOTBALL_DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==============================================================
# Endpoints API
# ==============================================================

@router.post("/exact_predict_fixture/{fixture_id}")
async def exact_predict_fixture(
    fixture_id: str,
    db: Session = Depends(get_football_db)
):
    """Predice un fixture (da DB) e restituisce raccomandazioni V3 con quote reali."""
    try:
        from app.db.models_football import Fixture
        from uuid import UUID

        # Normalizza fixture_id (UUID o stringa)
        try:
            if len(fixture_id) == 32 and '-' not in fixture_id:
                fixture_uuid = UUID(f"{fixture_id[:8]}-{fixture_id[8:12]}-{fixture_id[12:16]}-{fixture_id[16:20]}-{fixture_id[20:]}")
            elif len(fixture_id) == 36:
                fixture_uuid = UUID(fixture_id)
            else:
                fixture_uuid = fixture_id  # potrebbe essere chiave non-UUID
        except ValueError:
            fixture_uuid = fixture_id

        fixture = db.query(Fixture).filter(Fixture.id == fixture_uuid).first()
        if not fixture:
            raise HTTPException(status_code=404, detail=f"Fixture {fixture_id} not found")
        if not fixture.league_code:
            raise HTTPException(status_code=400, detail="Fixture missing league_code")

        predictor = _get_model_for_league(fixture.league_code)
        df = predictor.load_data(db, fixture.league_code)

        # Converte il fixture in riga e predice
        row = _build_fixture_row(fixture)
        extended_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        match_idx = len(extended_df) - 1
        prediction = predictor.predict_match(extended_df, match_idx)

        # Metadati
        prediction['fixture_id'] = fixture_id
        prediction['league_code'] = fixture.league_code
        prediction['fixture_date'] = fixture.match_date.isoformat() if getattr(fixture, 'match_date', None) else None
        prediction['note'] = 'Fixture prediction using V3 Complete system'

        clean_prediction = {k: _convert_numpy_types(v) for k, v in prediction.items()}

        # Quote reali per Multigol baseline
        real_quotes: Optional[Dict[str, float]] = None
        if fixture.avg_home_odds and fixture.avg_away_odds:
            real_quotes = {
                '1': fixture.avg_home_odds,
                '2': fixture.avg_away_odds,
                'X': fixture.avg_draw_odds or 3.2,
            }

        betting_recs = get_recommended_bets(clean_prediction, quotes=real_quotes)
        clean_recs = [_convert_numpy_types(r) for r in betting_recs]

        return {
            "success": True,
            "fixture_id": fixture_id,
            "league_code": fixture.league_code,
            "prediction": clean_prediction,
            "betting_recommendations": clean_recs,
            "model_info": {
                "version": "V3_COMPLETE",
                "data_source": "database",
                "training_matches": int(len(df)),
                "v3_features": "Aggressive thresholds + New Multigol markets",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exact_predict_match/{league_code}")
async def exact_predict_match(
    league_code: str,
    home_team: str,
    away_team: str,
    match_date: str | None = None,
    db: Session = Depends(get_football_db)
):
    """[LEGACY] Predice un match dato da nomi squadre (+/- data).
    Preferire /exact_predict_fixture in produzione.
    """
    try:
        predictor = _get_model_for_league(league_code)
        df = predictor.load_data(db, league_code)

        # Trova match storico se esiste
        if match_date:
            date_parsed = datetime.strptime(match_date, '%Y-%m-%d').date()
            matches = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team) & (df['Date'].dt.date == date_parsed)]
        else:
            matches = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)].tail(1)

        if len(matches) == 0:
            # Mock future fixture
            mock = {
                'Date': pd.to_datetime(match_date or datetime.utcnow().date()),
                'HomeTeam': home_team.strip(),
                'AwayTeam': away_team.strip(),
                'FTHG': np.nan,
                'FTAG': np.nan,
                'AvgH': 2.0,
                'AvgD': 3.2,
                'AvgA': 3.8,
                'Avg>2.5': 1.8,
                'Avg<2.5': 2.0,
            }
            extended_df = pd.concat([df, pd.DataFrame([mock])], ignore_index=True)
            match_idx = len(extended_df) - 1
            prediction = predictor.predict_match(extended_df, match_idx)
            prediction['note'] = 'Future fixture prediction using mock data'
        else:
            match_idx = matches.index[0]
            prediction = predictor.predict_match(df, match_idx)
            prediction['note'] = 'Historical match prediction'

        clean_prediction = {k: _convert_numpy_types(v) for k, v in prediction.items()}
        betting_recs = get_recommended_bets(clean_prediction)
        clean_recs = [_convert_numpy_types(r) for r in betting_recs]

        response = {
            "success": True,
            "prediction": clean_prediction,
            "betting_recommendations": clean_recs,
            "model_info": {
                "version": "EXACT_REPLICA",
                "data_source": "database",
                "matches_loaded": int(len(df)),
            },
        }
        return _convert_numpy_types(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all_fixtures_recommendations")
async def get_all_fixtures_recommendations(
    league_code: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_football_db),
    use_context: bool = True,
):
    """Raccomandazioni betting per i fixture in finestra temporale (± range standard)."""
    try:
        from app.db.models_football import Fixture
        from datetime import timedelta

        # Query base
        query = db.query(Fixture)
        if league_code:
            query = query.filter(Fixture.league_code == league_code)

        now = datetime.now()
        week_ago = now - timedelta(days=7)
        month_ahead = now + timedelta(days=30)

        fixtures = (
            query.filter(Fixture.match_date >= week_ago, Fixture.match_date <= month_ahead)
                 .order_by(Fixture.match_date)
                 .limit(limit)
                 .all()
        )
        if not fixtures:
            return {
                "success": True,
                "fixtures": [],
                "total_fixtures": 0,
                "message": "No fixtures found in the specified date range",
            }

        # Predictor cache per lega
        league_predictors: Dict[str, ExactSimpleFooballPredictor] = {}
        fixture_recommendations: List[Dict[str, Any]] = []
        processed_leagues: set[str] = set()

        for i, fixture in enumerate(fixtures, start=1):
            try:
                lc = fixture.league_code
                if lc not in league_predictors:
                    predictor = _get_model_for_league(lc)
                    df = predictor.load_data(db, lc)
                    predictor.data = df
                    league_predictors[lc] = predictor
                    processed_leagues.add(lc)

                predictor = league_predictors[lc]
                df = predictor.data

                row = _build_fixture_row(fixture)
                extended_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                
                from addons.compute_team_profiles import annotate_pre_match_elo
                
                try:
                    extended_df = annotate_pre_match_elo(extended_df)
                except Exception as e:
                    print(f"  - DEBUG: annotate_pre_match_elo failed for fixture {fixture.id}: {e}")
                    import traceback; traceback.print_exc()
                    raise  # opzionale: per non silenziarlo

                match_idx = len(extended_df) - 1

                try:
                    prediction = predictor.predict_match(extended_df, match_idx)
                    print(f"[{i}/{len(fixtures)}] Predicted fixture {fixture.id} ({fixture.home_team.name} vs {fixture.away_team.name})")
                except Exception as e:
                    print(f"  - DEBUG: Prediction failed for fixture {fixture.id}: {e}")
                    import traceback; traceback.print_exc()
                    raise  # opzionale: per non silenziarlo

                clean_prediction = _convert_numpy_types(prediction)

                real_quotes: Optional[Dict[str, float]] = None
                if fixture.avg_home_odds and fixture.avg_away_odds:
                    real_quotes = {
                        '1': fixture.avg_home_odds,
                        '2': fixture.avg_away_odds,
                        'X': fixture.avg_draw_odds or 3.2,
                    }
                print(f"  - Generating recommendations with real quotes: {real_quotes}")
                recs = get_recommended_bets(clean_prediction, quotes=real_quotes)
                clean_recs = [_convert_numpy_types(r) for r in recs]

                item = {
                    "fixture_id": str(fixture.id),
                    "match_date": fixture.match_date.isoformat() if getattr(fixture, 'match_date', None) else None,
                    "match_time": getattr(fixture, 'match_time', None),
                    "league_code": lc,
                    "home_team": fixture.home_team.name if getattr(fixture, 'home_team', None) else "Unknown",
                    "away_team": fixture.away_team.name if getattr(fixture, 'away_team', None) else "Unknown",
                    "odds_1x2": {
                        "home": fixture.avg_home_odds,
                        "draw": fixture.avg_draw_odds,
                        "away": fixture.avg_away_odds,
                        "over_25": fixture.avg_over_25_odds,
                        "under_25": fixture.avg_under_25_odds,
                    },
                    "betting_recommendations": clean_recs,
                    "total_recommendations": len(clean_recs),
                    "confidence_stats": {
                        "avg_confidence": round(sum(r['confidence'] for r in clean_recs) / len(clean_recs), 1) if clean_recs else 0,
                        "high_confidence": len([r for r in clean_recs if r['confidence'] >= 70]),
                        "medium_confidence": len([r for r in clean_recs if 60 <= r['confidence'] < 70]),
                    },
                }
                fixture_recommendations.append(item)

            except Exception as e:
                fixture_recommendations.append({
                    "fixture_id": str(getattr(fixture, 'id', 'unknown')),
                    "match_date": getattr(fixture, 'match_date', None),
                    "league_code": getattr(fixture, 'league_code', None),
                    "home_team": fixture.home_team.name if getattr(fixture, 'home_team', None) else "Unknown",
                    "away_team": fixture.away_team.name if getattr(fixture, 'away_team', None) else "Unknown",
                    "error": str(e),
                    "betting_recommendations": [],
                    "total_recommendations": 0,
                })
                continue

        total_recommendations = sum(f.get('total_recommendations', 0) for f in fixture_recommendations)
        successful_fixtures = len([f for f in fixture_recommendations if 'error' not in f])

        return {
            "success": True,
            "fixtures": fixture_recommendations,
            "total_fixtures": len(fixtures),
            "successful_predictions": successful_fixtures,
            "failed_predictions": len(fixtures) - successful_fixtures,
            "total_recommendations": total_recommendations,
            "processed_leagues": list(processed_leagues),
            "model_info": {
                "version": "V3_COMPLETE",
                "features": "Aggressive thresholds + New Multigol markets",
                "data_source": "database",
            },
            "filter_info": {
                "league_code": league_code,
                "date_range": f"{week_ago.strftime('%Y-%m-%d')} to {month_ahead.strftime('%Y-%m-%d')}",
                "limit": limit,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
