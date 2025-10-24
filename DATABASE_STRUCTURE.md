# Database Strutturato per Dati Calcistici

## Panoramica

Il sistema ora supporta il salvataggio dei dati CSV in un database strutturato con le seguenti entità:

## 🏛️ **Struttura Database**

### 📍 **Countries (Paesi)**
- `id`: UUID primario
- `code`: Codice paese a 3 lettere (es. "ITA", "ENG")
- `name`: Nome completo (es. "Italy", "England")  
- `flag_url`: URL icona bandiera
- `created_at`: Timestamp creazione

### 🏆 **Leagues (Leghe)**
- `id`: UUID primario
- `code`: Codice lega (es. "I1", "E0")
- `name`: Nome lega (es. "Serie A", "Premier League")
- `country_id`: Riferimento al paese
- `tier`: Livello (1=prima divisione, 2=seconda, etc.)
- `logo_url`: URL logo lega
- `active`: Se la lega è attiva
- `created_at`: Timestamp creazione

### 📅 **Seasons (Stagioni)**
- `id`: UUID primario
- `league_id`: Riferimento alla lega
- `name`: Nome stagione (es. "2023/2024")
- `code`: Codice stagione (es. "2324")
- `start_date`: Data prima partita (auto-calcolata)
- `end_date`: Data ultima partita (null se non conclusa)
- `is_completed`: True se stagione conclusa
- `total_matches`: Numero totale partite nel CSV
- `processed_matches`: Partite elaborate nel DB
- `csv_file_path`: Percorso file CSV originale
- `created_at/updated_at`: Timestamps

### 👥 **Teams (Squadre)**
- `id`: UUID primario
- `name`: Nome squadra originale
- `normalized_name`: Nome normalizzato per matching
- `country_id`: Riferimento al paese
- `logo_url`: URL logo squadra
- `founded_year`: Anno fondazione
- `created_at`: Timestamp creazione

### ⚽ **Matches (Partite)**
- `id`: UUID primario
- `season_id`: Riferimento alla stagione
- `home_team_id/away_team_id`: Riferimenti alle squadre
- `match_date`: Data partita
- `match_time`: Orario partita
- **Risultati:**
  - `home_goals_ft/away_goals_ft`: Gol tempo pieno (FTHG/FTAG)
  - `home_goals_ht/away_goals_ht`: Gol primo tempo (HTHG/HTAG)
- **Statistiche:**
  - `home_shots/away_shots`: Tiri totali (HS/AS)
  - `home_shots_target/away_shots_target`: Tiri in porta (HST/AST)
- **Quote:**
  - `avg_home_odds/avg_draw_odds/avg_away_odds`: Quote medie 1X2
  - `avg_over_25_odds/avg_under_25_odds`: Quote Over/Under 2.5 gol
- `csv_row_number`: Riferimento riga CSV originale
- `created_at`: Timestamp creazione

## 🔧 **Funzionalità API**

### ✅ **Download con Popolamento Database**
Tutti gli endpoint di download ora supportano il parametro `populate_db`:

```bash
# Download singolo con database
curl 'http://localhost:8000/csv/download-csv?league=I1&season=2324&populate_db=true'

# Download multiplo con database
curl -X POST 'http://localhost:8000/csv/download-multiple-seasons?league=I1&seasons=3&populate_db=true'

# Download massivo con database
curl -X POST 'http://localhost:8000/csv/download-all-recent?seasons=8&populate_db=true'
```

### 📊 **Risposta API Potenziata**
Le risposte includono ora informazioni sul database:

```json
{
  "success": true,
  "details": { ... },
  "database": {
    "success": true,
    "season_id": "uuid-stagione",
    "league": "Serie A (I1)",
    "season": "2023/2024",
    "matches_processed": 380,
    "total_rows": 380,
    "errors_count": 0
  }
}
```

## 🎯 **Caratteristiche Intelligenti**

### 📈 **Auto-Detection Stagione Conclusa**
- Calcola automaticamente se una stagione è conclusa
- Se l'ultima partita è > 90 giorni fa → `is_completed = true`
- Altrimenti → `is_completed = false` e `end_date = null`

### 🔄 **Gestione Duplicati**
- Partite esistenti vengono aggiornate invece che duplicate
- Matching basato su: stagione + data + squadre casa/trasferta

### 📝 **Normalizzazione Squadre**
- Nomi squadre normalizzati per matching consistente
- Rimozione caratteri speciali e spazi extra
- Conversione minuscolo per confronti

### 🗺️ **Mapping Colonne Intelligente**
- Gestisce CSV con nomi colonne diversi tra stagioni
- Mapping automatico: `BbAvH` → `AvgH`, `BbAv>2.5` → `Avg>2.5`
- Colonne mancanti → valori null invece di errori

## 🚀 **Vantaggi del Sistema**

### 📊 **Query Strutturate**
```sql
-- Partite Serie A 2023/24 con più di 3 gol
SELECT m.*, ht.name as home, at.name as away 
FROM matches m 
JOIN teams ht ON m.home_team_id = ht.id
JOIN teams at ON m.away_team_id = at.id
WHERE m.home_goals_ft + m.away_goals_ft > 3

-- Statistiche per paese
SELECT c.name, COUNT(DISTINCT l.id) as leagues_count
FROM countries c 
JOIN leagues l ON c.id = l.country_id
GROUP BY c.name
```

### 🔍 **Analisi Avanzate**
- Trend storici per squadra/lega
- Statistiche aggregate per stagione
- Confronti cross-league
- Analisi performance quote

### 💾 **Integrità Dati**
- Relazioni foreign key per consistenza
- Validazione automatica date/risultati
- Gestione errori granulare con log dettagliati
- Backup automatico tramite CSV originali

### 🔄 **Sincronizzazione**
- CSV e database sempre allineati
- Re-processing sicuro senza duplicati
- Aggiornamenti incrementali supportati

## 📋 **Status Implementazione**

✅ **Completato:**
- Modelli database completi
- Servizio popolamento dati
- Integrazione API download
- Gestione errori e logging
- Filtro colonne CSV potenziato

⏳ **In Corso:**
- Risoluzione import modelli
- Migrazione database
- Testing completo

🎯 **Prossimi Step:**
- Endpoint query database
- Dashboard statistiche
- API esportazione dati strutturati