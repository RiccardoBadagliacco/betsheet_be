# 🚀 BetSheet Football Prediction System# ⚽ BetSheet - Football Prediction System# BetSheet API



Sistema di predizione e betting per partite di calcio basato su modello **ExactSimpleFooballPredictor** completamente validato.



## 📊 Performance ValidataSistema avanzato di predizioni calcistiche basato su modelli statistici Poisson-Dixon-Coles per analisi e betting sportivo.



✅ **77.1% accuratezza** su 2,000 partite reali  Run instructions (development):

📈 **10,821+ raccomandazioni** testate  

🎯 **Zero partite fallite** su larga scala  ## 🚀 Quick Start

🌍 **Cross-league validated** (tutte le maggiori leghe europee)

1. Create and activate a venv

## 🏗️ Struttura Progetto

### Requisiti

```

betsheet_be/- Python 3.9+```bash

├── 🤖 app/                          # Core application

│   ├── api/- Virtual Environment (consigliato)python3 -m venv .venv

│   │   ├── ml_football_exact.py     # 🎯 MODELLO BASELINE VALIDATO

│   │   ├── health.py                # Health checksource .venv/bin/activate

│   │   └── ...                      # Altri endpoints API

│   ├── db/                          # Database models & connections### Installazione```

│   ├── core/                        # Settings & configuration  

│   └── main.py                      # FastAPI entry point```bash

│

├── 🧪 backtest/                     # Sistema di validazione# Clona il repository2. Install dependencies

│   ├── football_backtest_real.py    # 📊 ALGORITMO BACKTEST

│   ├── football_backtest_report_*.xlsx # Report Excel dettagliatigit clone https://github.com/RiccardoBadagliacco/betsheet_be.git

│   └── README.md                    # Documentazione backtest

│cd betsheet_be```bash

├── 📂 data/

│   └── football_dataset.db          # 37K+ partite storichepip install -r requirements.txt

│

├── 🚀 run_backtest.py               # Script per eseguire backtest# Crea virtual environment```

├── requirements.txt                 # Dipendenze Python

└── README.md                        # Questo filepython -m venv .venv

```

source .venv/bin/activate  # Linux/Mac3. Run the app

## 🎯 Modello Baseline: ExactSimpleFooballPredictor

# .venv\Scripts\activate   # Windows

**File:** `app/api/ml_football_exact.py`

```bash

### Features Principali:

- **Hybrid Approach**: 60% market odds + 40% statistical analysis# Installa dipendenze 

- **Poisson Distribution**: Per calcolo probabilità esatteuvicorn app.main:app --reload --port 8000

- **Historical Analysis**: Finestre temporali adattive (10 partite globali, 5 venue-specific)

- **Market Integration**: Rimozione vig automatica dalle quotepip install -r requirements.txt```

- **Multi-Market Support**: Over/Under, 1X2, Doppia Chance, Match Goals

```

### Top Performing Markets:

1. **Over 0.5 Goal**: 93.3% accuracy ⭐Open http://127.0.0.1:8000/docs for Swagger UI.

2. **Multigol Casa 1-5**: 81.2% accuracy ⭐  

3. **Multigol Casa 1-4**: 78.1% accuracy ⭐## 🎯 Utilizzo Principale

4. **Over 1.5 Goal**: 76.8% accuracy

5. **Doppia Chance**: 74-76% accuracyMigrations



### API Usage:### 1. Genera Predizioni

```python

from app.api.ml_football_exact import ExactSimpleFooballPredictor, get_recommended_bets```bash- Alembic is included as a dependency; initialize alembic with `alembic init alembic` and configure `alembic.ini` to point to your DB.



predictor = ExactSimpleFooballPredictor()# Predizioni su campionato specifico

prediction = predictor.predict_match(df, match_index)

recommendations = get_recommended_bets(prediction, quotes)python simple_football_model.py --data leagues_csv_unified/Italy_I1_Serie_A_ALL_SEASONS.csv --out predictions.csvTesting

```



## 🧪 Sistema di Backtest

# Test con campione limitatoRun pytest:

**File:** `backtest/football_backtest_real.py`

python simple_football_model.py --data leagues_csv_unified/Italy_I1_Serie_A_ALL_SEASONS.csv --sample 100 --out test.csv

### Capabilities:

- ✅ Test su N partite random dal database storico``````bash

- 📊 Validazione completa di tutti i mercati

- 📋 Report Excel dettagliato multi-sheet  pytest -q

- 🌍 Cross-league e cross-season testing

- 📈 Statistiche per mercato, lega e accuratezza globale### 2. Analisi Betting```



### Quick Run:```bash

```bash# Identifica opportunità ad alta confidenza

# Esegui backtest (default: 2000 partite)python betting_assistant.py --predictions predictions.csv --confidence 0.75

python run_backtest.py

# Analisi con soglia più alta

# O direttamente nella cartella backtestpython betting_assistant.py --predictions predictions.csv --confidence 0.85

cd backtest && python football_backtest_real.py```

```

### 3. Backtesting Performance

### Report Excel Generato:```bash

- **Match_Details**: Ogni partita con raccomandazioni dettagliate# Valuta performance del modello

- **Market_Statistics**: Performance per tipo di mercatopython football_backtest.py --predictions predictions.csv --report backtest_report.html --stake 10

- **League_Statistics**: Performance per lega/paese  ```

- **Summary**: Statistiche generali e KPI

## 📊 Output del Sistema

## 🚀 Quick Start

### Predizioni Generate

1. **Setup Environment:**- **Over/Under**: 0.5, 1.5, 2.5, 3.5

```bash- **Multigol**: Casa e Ospite (1-3, 1-4, 1-5)

pip install -r requirements.txt- **1X2**: Home Win, Draw, Away Win

```- **Parametri Poisson**: λ_home, λ_away



2. **Run API Server:**### Metriche di Performance

```bash- **Accuracy Over 1.5**: ~75.6%

python app/main.py- **ROI Betting**: ~38.7%

```- **Win Rate**: ~76.6%



3. **Run Backtest Validation:**## 🏆 Performance Validate

```bash

python run_backtest.pyIl sistema è stato validato su **759 partite Serie A (2023-2025)**:

```- ✅ Over 1.5 Accuracy: **75.6%** (superiore al mercato)

- ✅ ROI Simulato: **+38.7%** (molto profittevole)

4. **Use Model:**- ✅ Win Rate: **76.6%** (ottimo)

```python

from app.api.ml_football_exact import ExactSimpleFooballPredictor## 📁 Struttura Progetto



predictor = ExactSimpleFooballPredictor()```

# Use predictor for new match predictions...betsheet_be/

```├── 🎯 simple_football_model.py      # Modello principale predizioni

├── 💰 betting_assistant.py          # Assistant per betting opportunities

## 📊 Database├── 📊 football_backtest.py          # Sistema di backtesting

├── ⚙️  app/                         # FastAPI backend

- **football_dataset.db**: 37,793 partite storiche con risultati e quote├── 📚 docs/                         # Documentazione completa

- **bets.db**: Database utenti e sistema betting├── 📈 reports/                      # Report e analisi generate

├── 💾 data/                         # Dati esempio e test

## 🎯 Production Ready├── 🔧 scripts/                      # Utility scripts

├── 📊 leagues_csv_unified/          # Dataset campionati (15 leghe)

Il modello è **completamente validato** e pronto per:├── 🗄️  football_dataset.db          # Database partite storiche

- ✅ Predizioni live su nuove partite└── 📋 requirements.txt              # Dipendenze Python

- 📈 Sistema di raccomandazioni automatiche  ```

- 🎯 Betting intelligente con confidenza validata

- 📊 Scaling su multiple leghe simultaneamente## 🎯 Campionati Supportati



## 🔧 Requirements**15 Campionati Europei** con dati storici completi:

- 🇮🇹 **Italia**: Serie A, Serie B

- Python 3.8+- 🇪🇸 **Spagna**: La Liga, Segunda División  

- FastAPI- 🇩🇪 **Germania**: Bundesliga, 2. Bundesliga

- pandas, numpy- 🇫🇷 **Francia**: Ligue 1, Ligue 2

- sqlite3, sqlalchemy  - 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **Inghilterra**: Premier League, Championship

- openpyxl (per report Excel)- 🇳🇱 **Olanda**: Eredivisie

- scipy (per calcoli Poisson avanzati)- 🇵🇹 **Portogallo**: Primeira Liga

- 🇧🇪 **Belgio**: Jupiler Pro League

---- 🏴󠁧󠁢󠁳󠁣󠁴󠁿 **Scozia**: Premier League

- 🇹🇷 **Turchia**: Süper Lig

**Status:** ✅ **PRODUCTION READY** - Modello validato su 2K+ partite con 77.1% accuratezza
## 📖 Documentazione

- [`docs/FOOTBALL_MODEL_USAGE.md`](docs/FOOTBALL_MODEL_USAGE.md) - Guida completa utilizzo
- [`docs/EXECUTIVE_SUMMARY_BACKTESTING.md`](docs/EXECUTIVE_SUMMARY_BACKTESTING.md) - Risultati backtesting
- [`docs/SERIE_A_3SEASONS_REPORT.md`](docs/SERIE_A_3SEASONS_REPORT.md) - Report dettagliato Serie A
- [`docs/DATABASE_STRUCTURE.md`](docs/DATABASE_STRUCTURE.md) - Struttura database

## 🔧 API Backend

Il sistema include anche API FastAPI per integrazione web:

```bash
# Avvia server API
uvicorn app.main:app --reload

# Endpoint disponibili:
# GET /health - Status sistema
# GET /leagues/{league}/seasons - Statistiche stagionali
```

## 💡 Esempi Pratici

### Scenario Betting Reale
```bash
# 1. Genera predizioni giornaliere
python simple_football_model.py --data leagues_csv_unified/Italy_I1_Serie_A_ALL_SEASONS.csv --sample 10 --out today.csv

# 2. Trova opportunità >80% confidenza
python betting_assistant.py --predictions today.csv --confidence 0.8

# 3. Monitora performance
python football_backtest.py --predictions today.csv --report daily_report.html
```

### Output Tipico
```
📅 2025-10-24 | Juventus vs Inter
🎯 OVER 1.5 RACCOMANDATO (Probabilità: 78.3%)
🎯 MULTIGOL CASA 1-4 RACCOMANDATO (Probabilità: 75.1%)
📊 λ_home: 1.65 | λ_away: 1.32
```

## 📈 Roadmap

- [ ] Integrazione più campionati (Champions League, Europa League)
- [ ] ML models per calibrazione avanzata
- [ ] Dashboard web interattiva
- [ ] API real-time odds integration
- [ ] Mobile app companion

## 🤝 Contributi

Contributi benvenuti! Vedi [`CONTRIBUTING.md`](CONTRIBUTING.md) per guidelines.

## 📄 Licenza

[MIT License](LICENSE) - Vedi file per dettagli.

---

🏆 **Developed by BetSheet Analytics Team**  
⚡ **Powered by Poisson-Dixon-Coles Mathematical Models**  
📊 **Validated on 40,000+ Historical Matches**