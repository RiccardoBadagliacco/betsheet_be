# API Fixtures - Semplice

## Panoramica
API semplice per scaricare il palinsesto dal CSV fisso, pulirlo e salvarlo nella tabella `fixtures` del database.

## Endpoints

### 1. Scarica e Salva Fixtures
```
POST /api/v1/fixtures/download
```
- Scarica il CSV dall'URL fisso
- Pulisce i dati nel formato del progetto  
- Salva nella tabella `fixtures`
- Sostituisce fixtures esistenti

**Response:**
```json
{
  "success": true,
  "message": "Fixtures scaricate e salvate con successo",
  "total_downloaded": 100,
  "fixtures_saved": 85,
  "url": "https://www.football-data.co.uk/fixtures.csv"
}
```

### 2. Visualizza Fixtures
```
GET /api/v1/fixtures
```
- Recupera tutte le fixtures salvate nel database
- Ordinate per data e ora

**Response:**
```json
{
  "success": true,
  "fixtures": [
    {
      "id": "uuid",
      "match_date": "2025-10-25",
      "match_time": "15:00",
      "home_team_name": "Juventus",
      "away_team_name": "Napoli",
      "avg_home_odds": 2.10,
      "avg_draw_odds": 3.20,
      "avg_away_odds": 3.50,
      "downloaded_at": "2025-10-25 09:00:00"
    }
  ],
  "count": 1
}
```

### 3. Cancella Fixtures
```
DELETE /api/v1/fixtures
```
- Cancella tutte le fixtures dal database

### 4. Crea Tabelle
```
POST /api/v1/fixtures/create-tables
```
- Crea le tabelle del database (da usare solo una volta)

## Configurazione

L'URL del CSV è configurato in `app/api/fixtures.py`:
```python
FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"
```

## Utilizzo

```bash
# 1. Crea tabelle (solo prima volta)
curl -X POST "http://127.0.0.1:8000/api/v1/fixtures/create-tables"

# 2. Scarica fixtures
curl -X POST "http://127.0.0.1:8000/api/v1/fixtures/download"

# 3. Visualizza fixtures
curl -X GET "http://127.0.0.1:8000/api/v1/fixtures"
```

## Formato Dati

La tabella `fixtures` ha lo stesso formato di `matches`:
- `match_date`, `match_time` - Data e ora partita
- `home_team_name`, `away_team_name` - Squadre (nomi originali dal CSV)
- `avg_home_odds`, `avg_draw_odds`, `avg_away_odds` - Quote
- `avg_over_25_odds`, `avg_under_25_odds` - Quote over/under
- Tutti i campi risultato sono NULL (partite da giocare)