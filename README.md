# ⚽ BetSheet - Football Prediction System# BetSheet API



Sistema avanzato di predizioni calcistiche basato su modelli statistici Poisson-Dixon-Coles per analisi e betting sportivo.

Run instructions (development):

## 🚀 Quick Start

1. Create and activate a venv

### Requisiti

- Python 3.9+```bash

- Virtual Environment (consigliato)python3 -m venv .venv

source .venv/bin/activate

### Installazione```

```bash

# Clona il repository2. Install dependencies

git clone https://github.com/RiccardoBadagliacco/betsheet_be.git

cd betsheet_be```bash

pip install -r requirements.txt

# Crea virtual environment```

python -m venv .venv

source .venv/bin/activate  # Linux/Mac3. Run the app

# .venv\Scripts\activate   # Windows

```bash

# Installa dipendenzeuvicorn app.main:app --reload --port 8000

pip install -r requirements.txt```

```

Open http://127.0.0.1:8000/docs for Swagger UI.

## 🎯 Utilizzo Principale

Migrations

### 1. Genera Predizioni

```bash- Alembic is included as a dependency; initialize alembic with `alembic init alembic` and configure `alembic.ini` to point to your DB.

# Predizioni su campionato specifico

python simple_football_model.py --data leagues_csv_unified/Italy_I1_Serie_A_ALL_SEASONS.csv --out predictions.csvTesting



# Test con campione limitatoRun pytest:

python simple_football_model.py --data leagues_csv_unified/Italy_I1_Serie_A_ALL_SEASONS.csv --sample 100 --out test.csv

``````bash

pytest -q

### 2. Analisi Betting```

```bash
# Identifica opportunità ad alta confidenza
python betting_assistant.py --predictions predictions.csv --confidence 0.75

# Analisi con soglia più alta
python betting_assistant.py --predictions predictions.csv --confidence 0.85
```

### 3. Backtesting Performance
```bash
# Valuta performance del modello
python football_backtest.py --predictions predictions.csv --report backtest_report.html --stake 10
```

## 📊 Output del Sistema

### Predizioni Generate
- **Over/Under**: 0.5, 1.5, 2.5, 3.5
- **Multigol**: Casa e Ospite (1-3, 1-4, 1-5)
- **1X2**: Home Win, Draw, Away Win
- **Parametri Poisson**: λ_home, λ_away

### Metriche di Performance
- **Accuracy Over 1.5**: ~75.6%
- **ROI Betting**: ~38.7%
- **Win Rate**: ~76.6%

## 🏆 Performance Validate

Il sistema è stato validato su **759 partite Serie A (2023-2025)**:
- ✅ Over 1.5 Accuracy: **75.6%** (superiore al mercato)
- ✅ ROI Simulato: **+38.7%** (molto profittevole)
- ✅ Win Rate: **76.6%** (ottimo)

## 📁 Struttura Progetto

```
betsheet_be/
├── 🎯 simple_football_model.py      # Modello principale predizioni
├── 💰 betting_assistant.py          # Assistant per betting opportunities
├── 📊 football_backtest.py          # Sistema di backtesting
├── ⚙️  app/                         # FastAPI backend
├── 📚 docs/                         # Documentazione completa
├── 📈 reports/                      # Report e analisi generate
├── 💾 data/                         # Dati esempio e test
├── 🔧 scripts/                      # Utility scripts
├── 📊 leagues_csv_unified/          # Dataset campionati (15 leghe)
├── 🗄️  football_dataset.db          # Database partite storiche
└── 📋 requirements.txt              # Dipendenze Python
```

## 🎯 Campionati Supportati

**15 Campionati Europei** con dati storici completi:
- 🇮🇹 **Italia**: Serie A, Serie B
- 🇪🇸 **Spagna**: La Liga, Segunda División  
- 🇩🇪 **Germania**: Bundesliga, 2. Bundesliga
- 🇫🇷 **Francia**: Ligue 1, Ligue 2
- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **Inghilterra**: Premier League, Championship
- 🇳🇱 **Olanda**: Eredivisie
- 🇵🇹 **Portogallo**: Primeira Liga
- 🇧🇪 **Belgio**: Jupiler Pro League
- 🏴󠁧󠁢󠁳󠁣󠁴󠁿 **Scozia**: Premier League
- 🇹🇷 **Turchia**: Süper Lig

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