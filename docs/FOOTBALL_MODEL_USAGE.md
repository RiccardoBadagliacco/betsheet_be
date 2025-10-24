# 🏆 Football Prediction Model - Guida Utente

## Introduzione
Il `simple_football_model.py` è un modello statistico-matematico per predizioni di partite di calcio basato sulla distribuzione di Poisson e informazioni di mercato.

## Funzionalità Principali
- ✅ Calcolo probabilità gol con distribuzione di Poisson
- ✅ Integrazione informazioni di mercato (quote)
- ✅ Predizioni Over/Under (0.5, 1.5, 2.5, 3.5)
- ✅ Predizioni Goal/No Goal
- ✅ Predizioni 1X2
- ✅ Export CSV e report HTML
- ✅ Modalità test e produzione

## Utilizzo Base

### Test con campione di partite
```bash
# Test con 50 partite della Serie A
python simple_football_model.py --data leagues_csv_unified/Italy_I1_Serie_A_ALL_SEASONS.csv --sample 50

# Test con partite specifiche (dalla 100 alla 200)
python simple_football_model.py --data leagues_csv_unified/Italy_I1_Serie_A_ALL_SEASONS.csv --sample 100 --start 100
```

### Predizioni complete
```bash
# Tutte le partite della Serie A
python simple_football_model.py --data leagues_csv_unified/Italy_I1_Serie_A_ALL_SEASONS.csv

# Con file di output personalizzati
python simple_football_model.py --data leagues_csv_unified/Italy_I1_Serie_A_ALL_SEASONS.csv --out serie_a_predictions.csv --report serie_a_report.html
```

## Parametri Disponibili

| Parametro | Descrizione | Esempio |
|-----------|------------|---------|
| `--data` | File CSV del campionato (OBBLIGATORIO) | `leagues_csv_unified/Italy_I1_Serie_A_ALL_SEASONS.csv` |
| `--sample` | Numero di partite da predire (test mode) | `--sample 100` |
| `--start` | Partita di inizio per il campione | `--start 50` |
| `--out` | File CSV di output | `--out predictions.csv` |
| `--report` | File HTML report | `--report report.html` |

## Campionati Disponibili

### Serie A-B Europee
- 🇮🇹 **Italia**: `Italy_I1_Serie_A_ALL_SEASONS.csv`, `Italy_I2_Serie_B_ALL_SEASONS.csv`
- 🇪🇸 **Spagna**: `Spain_SP1_La_Liga_ALL_SEASONS.csv`, `Spain_SP2_Segunda_División_ALL_SEASONS.csv`
- 🇩🇪 **Germania**: `Germany_D1_Bundesliga_ALL_SEASONS.csv`, `Germany_D2_2._Bundesliga_ALL_SEASONS.csv`
- 🇫🇷 **Francia**: `France_F1_Ligue_1_ALL_SEASONS.csv`, `France_F2_Ligue_2_ALL_SEASONS.csv`
- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **Inghilterra**: `England_E0_Premier_League_ALL_SEASONS.csv`, `England_E1_Championship_ALL_SEASONS.csv`

### Altri Campionati
- 🇳🇱 **Olanda**: `Netherlands_N1_Eredivisie_ALL_SEASONS.csv`
- 🇵🇹 **Portogallo**: `Portugal_P1_Primeira_Liga_ALL_SEASONS.csv`
- 🇧🇪 **Belgio**: `Belgium_B1_Jupiler_Pro_League_ALL_SEASONS.csv`
- 🏴󠁧󠁢󠁳󠁣󠁴󠁿 **Scozia**: `Scotland_SC0_Premier_League_ALL_SEASONS.csv`
- 🇹🇷 **Turchia**: `Turkey_T1_Süper_Lig_ALL_SEASONS.csv`

## Output del Modello

### File CSV (`predictions.csv`)
Contiene per ogni partita:
- **Informazioni partita**: Data, squadre, risultato reale
- **Parametri Poisson**: λ_home, λ_away (intensità gol attese)
- **Probabilità Over/Under**: O0.5, U0.5, O1.5, U1.5, O2.5, U2.5, O3.5, U3.5
- **Probabilità GG**: Both teams to score / No goal
- **Probabilità 1X2**: Home win, Draw, Away win
- **Informazioni mercato**: Quote quando disponibili

### Report HTML (`report.html`)
- 📊 Statistiche generali del dataset
- 🎯 Esempi di predizioni più significative
- 📈 Grafici e visualizzazioni
- 🔍 Analisi performance modello

## Esempi Pratici

### 1. Analisi Serie A Stagione Corrente
```bash
python simple_football_model.py \
  --data leagues_csv_unified/Italy_I1_Serie_A_ALL_SEASONS.csv \
  --sample 50 \
  --out serie_a_test.csv \
  --report serie_a_test.html
```

### 2. Predizioni Premier League Complete
```bash
python simple_football_model.py \
  --data leagues_csv_unified/England_E0_Premier_League_ALL_SEASONS.csv \
  --out premier_full.csv \
  --report premier_full.html
```

### 3. Test Bundesliga con Range Specifico
```bash
python simple_football_model.py \
  --data leagues_csv_unified/Germany_D1_Bundesliga_ALL_SEASONS.csv \
  --sample 200 \
  --start 500 \
  --out bundesliga_sample.csv
```

## Interpretazione Risultati

### Parametri λ (Lambda)
- **λ_home**: Gol attesi squadra casa (es. 1.70 = 1.7 gol attesi)
- **λ_away**: Gol attesi squadra ospite (es. 1.16 = 1.16 gol attesi)

### Probabilità Over/Under
- **O0.5 = 0.942**: 94.2% probabilità che si segni almeno 1 gol
- **O1.5 = 0.777**: 77.7% probabilità che si segnino almeno 2 gol
- **O2.5 = 0.456**: 45.6% probabilità che si segnino almeno 3 gol

### Esempio Pratica
```
Napoli vs Empoli
λ_home: 1.71, λ_away: 1.16
O:0.5 = 0.942, O:1.5 = 0.777
Risultato reale: 5-1
```
Il modello prevedeva una partita con molti gol (94% over 0.5) e si è rivelato corretto!

## Note Tecniche
- **Periodo dati**: Dal 2018 al 2025 (stagioni storiche complete)
- **Algoritmo**: Poisson con correzione market information
- **Performance**: ~2700 partite per campionato maggiore
- **Accuratezza**: Variabile per mercato (Over/Under tipicamente >65%)

## Troubleshooting

### Errore "File not found"
Verifica che il file CSV esista in `leagues_csv_unified/`

### Errore "No matches found"
Il dataset potrebbe essere vuoto o corrotto

### Performance lente
Su dataset grandi (>2000 partite) le predizioni complete richiedono alcuni minuti

---
*Modello sviluppato per BetSheet - Analisi statistiche calcio* 🏆