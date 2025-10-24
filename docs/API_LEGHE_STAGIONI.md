# 🏆 API Leghe e Stagioni - Documentazione

## 📊 **Panoramica**

Ho creato un'API completa che ritorna tutte le leghe con la lista delle stagioni disponibili nel database football. L'API offre tre endpoint differenti per diverse necessità.

## 🎯 **Endpoint Principali**

### 1. `/api/v1/leagues/seasons` - **Formato Raggruppato per Paese** ⭐

**Metodo:** `GET`  
**Descrizione:** Ritorna tutte le leghe raggruppate per paese con stagioni ordinate

**Risposta:**
```json
{
  "countries": [
    {
      "country": "Italy",
      "leagues_count": 2,
      "leagues": [
        {
          "code": "I1",
          "name": "Serie A",
          "seasons": ["2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819"]
        },
        {
          "code": "I2", 
          "name": "Serie B",
          "seasons": ["2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819"]
        }
      ]
    }
  ],
  "total_countries": 10,
  "total_leagues": 15,
  "total_seasons": 118
}
```

## 🆕 **Nuove API Dettagliate**

### 4. `/api/v1/leagues/league/{league_id}/seasons-detailed` - **Stagioni Dettagliate per ID Lega**

**Metodo:** `GET`  
**Descrizione:** Ritorna tutte le stagioni di una lega con dati dettagliati e statistiche

**Parametri:**
- `league_id` (UUID): ID della lega nel database

**Risposta:**
```json
{
  "league": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "code": "I1",
    "name": "Serie A",
    "country": "Italy",
    "tier": 1,
    "active": true,
    "created_at": "2024-10-24T10:00:00"
  },
  "seasons": [
    {
      "id": "660e8400-e29b-41d4-a716-446655440000",
      "code": "2526",
      "name": "2025/2026",
      "start_date": "2025-08-15",
      "end_date": null,
      "is_completed": false,
      "statistics": {
        "expected_matches": 380,
        "actual_matches": 95,
        "processed_matches": 95,
        "completion_percentage": 25.0
      },
      "dates": {
        "created_at": "2024-10-24T10:00:00",
        "updated_at": "2024-10-24T15:30:00"
      }
    }
  ],
  "statistics": {
    "total_seasons": 8,
    "completed_seasons": 7,
    "active_seasons": 1,
    "total_matches": 2847,
    "avg_matches_per_season": 355.9
  }
}
```

### 5. `/api/v1/leagues/league/code/{league_code}/seasons-detailed` - **Stagioni Dettagliate per Codice Lega**

**Metodo:** `GET`  
**Descrizione:** Come l'endpoint precedente ma usa il codice lega invece dell'ID

**Parametri:**
- `league_code` (string): Codice della lega (es: I1, E0, D1)

**Esempio:** `/api/v1/leagues/league/code/I1/seasons-detailed`

### 6. `/api/v1/leagues/season/{season_id}/details` - **Dettagli Stagione Specifica**

**Metodo:** `GET`  
**Descrizione:** Informazioni dettagliate per una singola stagione

**Parametri:**
- `season_id` (UUID): ID della stagione
- `include_matches` (boolean, optional): Include lista partite (default: false)

**Risposta Base:**
```json
{
  "season": {
    "id": "660e8400-e29b-41d4-a716-446655440000",
    "code": "2526",
    "name": "2025/2026",
    "start_date": "2025-08-15",
    "end_date": null,
    "is_completed": false,
    "csv_file_path": "/path/to/csv",
    "dates": {
      "created_at": "2024-10-24T10:00:00",
      "updated_at": "2024-10-24T15:30:00",
      "first_match": "2025-08-15",
      "last_match": "2025-10-20"
    }
  },
  "league": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "code": "I1",
    "name": "Serie A",
    "country": "Italy",
    "tier": 1
  },
  "statistics": {
    "expected_matches": 380,
    "actual_matches": 95,
    "completed_matches": 85,
    "processed_matches": 95,
    "completion_percentage": 25.0,
    "played_percentage": 89.5
  }
}
```

**Con `include_matches=true`:**
```json
{
  "season": { /* ... */ },
  "league": { /* ... */ },
  "statistics": { /* ... */ },
  "matches": {
    "data": [
      {
        "id": "770e8400-e29b-41d4-a716-446655440000",
        "date": "2025-10-20",
        "time": "20:45",
        "home_team": "Juventus",
        "away_team": "Inter Milan",
        "score_ft": "2-1",
        "score_ht": "1-0",
        "csv_row": 95
      }
    ],
    "total_shown": 100,
    "note": "Showing up to 100 most recent matches"
  }
}
```

L'API è stata estesa con funzionalità dettagliate e ora supporta anche il raggruppamento per paese! 🎉
```

### 2. `/api/v1/leagues/simple` - **Formato Semplificato**

**Metodo:** `GET`  
**Descrizione:** Formato minimalista con solo codici lega come chiavi

**Risposta:**
```json
{
  "I1": ["2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819"],
  "E0": ["2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819"],
  "D1": ["2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819"],
  "SP1": ["2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819"]
}
```

### 3. `/api/v1/leagues/seasons/{league_code}` - **Lega Specifica**

**Metodo:** `GET`  
**Parametro:** `league_code` (es: I1, E0, D1)  
**Descrizione:** Stagioni per una lega specifica

**Esempio:** `/api/v1/leagues/seasons/I1`

**Risposta:**
```json
{
  "league": {
    "code": "I1",
    "name": "Serie A", 
    "country": "Italy"
  },
  "seasons": ["2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819"],
  "total_seasons": 8
}
```

### 4. `/api/v1/leagues/league-code/{league_code}/seasons-detailed` - **Stagioni Dettagliate** 🆕

**Metodo:** `GET`  
**Parametro:** `league_code` (es: I1, E0, D1)  
**Descrizione:** Stagioni con dati statistici completi per una lega specifica

**Esempio:** `/api/v1/leagues/league-code/I1/seasons-detailed`

**Risposta:**
```json
{
  "league": {
    "id": "uuid-here",
    "code": "I1",
    "name": "Serie A",
    "tier": 1
  },
  "seasons": [
    {
      "id": "season-uuid",
      "code": "2526",
      "name": "2025/2026", 
      "start_date": "2025-08-15",
      "end_date": null,
      "is_completed": false,
      "matches": {
        "total_expected": 380,
        "processed": 45,
        "actual_in_db": 45,
        "first_match_date": "2025-08-15",
        "last_match_date": "2025-10-20"
      },
      "statistics": {
        "teams_count": 20,
        "avg_goals_per_match": 2.85
      }
    }
  ],
  "total_seasons": 8
}
```

### 5. `/api/v1/leagues/season/{season_id}` - **Dettagli Stagione** 🆕

**Metodo:** `GET`  
**Parametro:** `season_id` (UUID della stagione)  
**Query Param:** `include_matches=true` (opzionale)  
**Descrizione:** Dati dettagliati di una stagione specifica

**Esempio:** `/api/v1/leagues/season/{season_id}?include_matches=true`

**Risposta:**
```json
{
  "id": "season-uuid",
  "code": "2526",
  "name": "2025/2026",
  "start_date": "2025-08-15", 
  "end_date": null,
  "is_completed": false,
  "league": {
    "id": "league-uuid",
    "code": "I1", 
    "name": "Serie A",
    "tier": 1
  },
  "matches": {
    "total_expected": 380,
    "processed": 45,
    "actual_in_db": 45,
    "first_match_date": "2025-08-15",
    "last_match_date": "2025-10-20"
  },
  "statistics": {
    "teams_count": 20,
    "avg_total_goals_per_match": 2.85,
    "avg_home_goals_per_match": 1.65,
    "avg_away_goals_per_match": 1.20,
    "max_goals_in_match": 8,
    "min_goals_in_match": 0
  },
  "recent_matches": [
    {
      "id": "match-uuid",
      "date": "2025-10-20",
      "time": "15:00",
      "home_team": "Juventus",
      "away_team": "Inter",
      "home_goals": 2,
      "away_goals": 1,
      "total_goals": 3
    }
  ]
}
```

## 🗄️ **Struttura Database**

L'API interroga il database football con la seguente struttura:

```
leagues (15 record)
├── id (UUID)
├── code (I1, E0, D1, etc.)
├── name (Serie A, Premier League, etc.)
└── country_id

seasons (118 record)  
├── id (UUID)
├── league_id (FK to leagues)
├── code (2526, 2425, etc.)
├── name (2025/2026, 2024/2025, etc.)
└── is_completed

matches (37,793 record)
├── season_id (FK to seasons)
├── match_date
└── [altri campi partita]
```

## ⚡ **Caratteristiche**

### 🎯 **Ottimizzazioni**
- Query JOIN ottimizzate tra tabelle `leagues` e `seasons`
- Ordinamento stagioni dalla più recente alla più vecchia
- Gestione errori completa con HTTPException
- Solo leghe con dati reali (non mostra leghe vuote)

### 📊 **Dati Attuali** 
- **15 leghe** europee supportate
- **118 stagioni** totali nel database
- **37,793 partite** caricate
- **8 stagioni** per lega in media

### 🌍 **Leghe Supportate**
| Codice | Lega | Paese |
|--------|------|--------|
| I1 | Serie A | Italy |
| E0 | Premier League | England |  
| D1 | Bundesliga | Germany |
| SP1 | La Liga | Spain |
| F1 | Ligue 1 | France |
| N1 | Eredivisie | Netherlands |
| P1 | Primeira Liga | Portugal |
| B1 | Jupiler Pro League | Belgium |
| T1 | Süper Lig | Turkey |
| SC0 | Premier League | Scotland |
| E1 | Championship | England |
| D2 | 2. Bundesliga | Germany |
| I2 | Serie B | Italy |
| SP2 | Segunda División | Spain |
| F2 | Ligue 2 | France |

## 🛠️ **Esempi di Utilizzo**

### JavaScript/Frontend
```javascript
// Ottieni tutte le leghe raggruppate per paese
const countriesData = await fetch('/api/v1/leagues/seasons').then(r => r.json());

// Formato semplice per select/dropdown
const simpleLeagues = await fetch('/api/v1/leagues/simple').then(r => r.json());

// Stagioni per Serie A (semplice)
const serieASeasons = await fetch('/api/v1/leagues/seasons/I1').then(r => r.json());

// Stagioni Serie A con dati dettagliati
const serieADetailed = await fetch('/api/v1/leagues/league-code/I1/seasons-detailed').then(r => r.json());

// Dettagli di una stagione specifica
const seasonDetails = await fetch(`/api/v1/leagues/season/${seasonId}?include_matches=true`).then(r => r.json());
```

### Python
```python
import requests

# Tutte le leghe
response = requests.get('http://localhost:8000/api/v1/leagues/seasons')
leagues_data = response.json()

# Una lega specifica  
serie_a = requests.get('http://localhost:8000/api/v1/leagues/seasons/I1').json()
```

### cURL
```bash
# Tutte le leghe raggruppate per paese
curl -X GET "http://localhost:8000/api/v1/leagues/seasons"

# Formato semplice
curl -X GET "http://localhost:8000/api/v1/leagues/simple" 

# Serie A (semplice)
curl -X GET "http://localhost:8000/api/v1/leagues/seasons/I1"

# Serie A con dati dettagliati
curl -X GET "http://localhost:8000/api/v1/leagues/league-code/I1/seasons-detailed"

# Dettagli stagione specifica con partite
curl -X GET "http://localhost:8000/api/v1/leagues/season/{season-uuid}?include_matches=true"
```

## 🚀 **Integrazione**

L'API è integrata nel router principale di FastAPI:

- **Router:** `leagues_seasons.router`
- **Prefix:** `/api/v1`  
- **Tags:** `["Leagues & Seasons"]`
- **Database:** Football database separato
- **Dependency:** `get_football_db()`

## 📈 **Vantaggi**

1. **Flessibilità:** Tre formati diversi per diverse esigenze
2. **Performance:** Query ottimizzate con JOIN specifici  
3. **Completezza:** Informazioni dettagliate su leghe e paesi
4. **Affidabilità:** Dati reali dal database, non costanti statiche
5. **Scalabilità:** Struttura estendibile per nuove leghe/stagioni

L'API è pronta per l'uso e fornisce tutti i dati necessari per costruire interfacce che mostrano leghe e stagioni disponibili! 🎯