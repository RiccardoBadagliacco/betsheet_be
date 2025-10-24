# 💾 Data Directory

Questa cartella è destinata ai file di dati del sistema BetSheet.

## 📂 Struttura Dati

- **Database SQLite** (da posizionare qui): `football_dataset.db`, `bets.db`
- **CSV Predizioni**: Output del modello di predizione
- **CSV Storici**: Dati delle partite per backtesting
- **Cache Files**: File temporanei di elaborazione

## 🎯 File Principali

```
data/
├── football_dataset.db     # Database principale (40k+ partite) ✅ SPOSTATO
├── bets.db                # Database scommesse ✅ SPOSTATO
├── predictions_YYYYMMDD.csv  # Predizioni giornaliere
└── cache/                 # File temporanei
```

## 📋 Note

- Questa cartella è monitorata da `.gitignore` per i file di dati sensibili
- I database devono essere inizializzati con `scripts/init_football_db.py`
- I CSV di output vengono generati automaticamente dal modello

## 🚀 Setup Iniziale

1. ✅ Database `bets.db` e `football_dataset.db` spostati in questa cartella
2. Esegui `python scripts/init_football_db.py` per inizializzare il database football
3. Verifica che i path nei script puntino a `data/`