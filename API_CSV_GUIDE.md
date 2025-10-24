# API CSV Download - Guida Rapida

## Endpoint Principali

### 🚀 Download Automatico Completo
Scarica le ultime 6 stagioni per tutte le leghe:
```bash
curl -X POST 'http://localhost:8000/csv/download-all-recent'
```

### 🎯 Download per Leghe Specifiche  
Scarica le ultime 3 stagioni solo per le top 5 leghe europee:
```bash
curl -X POST 'http://localhost:8000/csv/download-all-recent?seasons=3&league_filter=E0,D1,I1,SP1,F1'
```

### 🏆 Download per Singola Lega
Scarica le ultime 6 stagioni della Serie A:
```bash
curl -X POST 'http://localhost:8000/csv/download-multiple-seasons?league=I1&seasons=6'
```

### 📅 Stagioni Personalizzate
Scarica stagioni specifiche per la Premier League:
```bash
curl -X POST 'http://localhost:8000/csv/download-multiple-seasons?league=E0&custom_seasons=2324,2223,2122'
```

## Endpoint di Utilità

### 📋 Lista Leghe Supportate
```bash
curl 'http://localhost:8000/csv/leagues'
```

### 📂 File Scaricati per Lega
```bash
curl 'http://localhost:8000/csv/leagues/I1/files'
```

### ❓ Help Completo
```bash
curl 'http://localhost:8000/csv/help'
```

## Codici Leghe Principali
- **E0**: Premier League (England)
- **I1**: Serie A (Italy)  
- **D1**: Bundesliga (Germany)
- **SP1**: La Liga (Spain)
- **F1**: Ligue 1 (France)

## Struttura File
```
leagues/
├── I1/2023_2024.csv
├── E0/2024_2025.csv  
└── D1/2025_2026.csv
```

Le API gestiscono automaticamente errori, retry e organizzazione dei file!