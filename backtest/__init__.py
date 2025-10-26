"""
🧪 Football Betting Backtest System

Sistema di backtest per validare le performance del modello ExactSimpleFooballPredictor
su dati storici reali con report Excel dettagliati.

Classes:
    FootballBacktest: Sistema di backtest completo

Functions:
    main(): Entry point per eseguire backtest su 2000 partite
"""

from .football_backtest_real import FootballBacktest

__version__ = "1.0.0"
__author__ = "BetSheet Team"
__all__ = ["FootballBacktest"]