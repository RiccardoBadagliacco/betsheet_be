# 🔧 Scripts & Utilities

Questa cartella contiene gli script di utilità per il sistema BetSheet.

## 📜 Script Disponibili

### 🏗️ Inizializzazione Sistema
- **`init_football_db.py`** - Inizializza database football con dati storici

### 📊 Generazione Dati  
- **`generate_league_csvs.py`** - Genera CSV per tutte le 15 leghe europee

## 🚀 Utilizzo Script

### Setup Iniziale Database
```bash
cd /path/to/betsheet_be
python scripts/init_football_db.py
```

### Generazione CSV Leghe
```bash
python scripts/generate_league_csvs.py
```

## 📋 Note Tecniche

- Gli script assumono la struttura di cartelle organizzata
- Database e output vengono salvati in `data/`
- Verificare sempre i path relativi prima dell'esecuzione
- Script progettati per essere eseguiti dalla root del progetto

## 🔍 Dipendenze

Tutti gli script richiedono:
- Python 3.8+
- Dipendenze da `requirements.txt`
- Database inizializzati correttamente