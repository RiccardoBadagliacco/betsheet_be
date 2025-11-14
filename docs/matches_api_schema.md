# Schema `matches_api` (esempio 0a0a1c00-11ac-4708-bbb1-7a480bd6ffb6)

File di riferimento: `data/matches_api/0a0a1c00-11ac-4708-bbb1-7a480bd6ffb6.json`

Questo JSON è il payload che l’API espone per ogni match e contiene quattro blocchi principali:

1. `match`: metadati della partita presi dal DB `matches`.
2. `stats`: statistiche formattate dal servizio `get_team_stats`.
3. `picchetti`: picchetto tecnico calcolato in `app/analytics/picchetto_tecnico.py`.
4. `metrics`: sintesi aggiuntive (oggi valorizzate solo per il mercato 1X2).

Di seguito il dettaglio campo per campo e come vengono calcolati.

---

## 1. Blocco `match`

| Campo | Significato / Calcolo |
| --- | --- |
| `id` | UUID della partita (`Match.id`). |
| `date`, `time` | Data e ora locale (`Match.match_date`, `match_time`). |
| `home_team`, `away_team` | Oggetti con `id` e `name` delle squadre (`Team.id`/`Team.name`). |
| `result.home_ft`, `result.away_ft` | Gol finali (`Match.home_goals_ft`, `away_goals_ft`). |
| `odds` | Quote medie 1X2 (home/draw/away) dai campi `avg_home_odds`, `avg_draw_odds`, `avg_away_odds`. |
| `odds_ou_25` | Quote medie Over/Under 2.5 (`avg_over_25_odds`, `avg_under_25_odds`). |
| `league` | Codice, nome e paese della lega (da `Match.season.league`). |
| `season` | UUID e nome della stagione (`Season`). |

Qualsiasi campo di quota può essere `null` se mancano dati nella riga del database.

---

## 2. Blocco `stats`

Struttura: `stats.<side>.<market>.<sezione>.stats.<metrica>`.

### Origine dati
Vengono da `app/analytics/get_team_stats_service.py`, che aggrega il database eventi per mercato/squadra, e poi passano per `format_stats_for_side`. Ogni metrica viene esposta come:

```json
"partite": {"value": 3, "label": "N"}
```

#### Sezioni disponibili
- `totali`: stagione corrente.
- `recenti`: ultime `n` gare (default 5).
- `totali_home` / `totali_away`: solo partite giocate in casa/trasferta.
- `recenti_home` / `recenti_away`: ultime `n` partite per lato di campo.

#### Mercati
1. `1X2`: metriche `vittorie`, `pareggi`, `sconfitte`.
2. `OU25` e `OU15`: metriche `under`, `over` sul numero di gol totali.
3. `GNG`: metriche `goal`, `no_goal`.

I valori sono semplici conteggi; nell’esempio `home.OU25.totali.stats.over.value = 2` indica 2 over 2.5 su 3 partite stagionali del Beerschot.

---

## 3. Blocco `picchetti`

Generato da `calcola_picchetto_*` in `app/analytics/picchetto_tecnico.py`, usando:

- Le statistiche formattate.
- Le quote bookmaker (se presenti) per stimare l’allibramento.

### Campi comuni
| Campo | Come si ottiene |
| --- | --- |
| `probabilità_%` | Percentuale stimata dal modello. Per il mercato 1X2, il modello combina quattro contesti (totali/recenti/home-away) e calcola la media delle probabilità di vittoria/pareggio/sconfitta. Per i mercati binari (OU, GNG) usa `_calcola_picchetto_binario`, che pesa le frequenze di under/over o goal/no-goal. |
| `quota_reale` | `100 / probabilità_%` (1X2) o `1 / probabilità_decimale` (mercati binari). Esprime la quota fair senza margine del bookmaker. |
| `quota_bookmaker` | Le quote ricevute in ingresso (`match.odds`, `match.odds_ou_25`, eventuali altre quote). |
| `allibramento%` | Somma delle probabilità implicite del bookmaker (`100 / quota`) meno 100. È il margine totale. |
| `spalmatura_allibramento%` | Peso del margine per ogni segno: `(probabilità_implicita / somma_implicite) * 100`. |
| `quota_reale_allibrata` | Applicazione del margine sul modello (funzione `allibra_quote_per_spalmatura`). Assente quando mancano le quote bookmaker, come per OU1.5/GG nell’esempio. |
| `analisi` | Commento sintetico generato da `genera_commento_picchetto*`, che riassume favorito, value bet e allibramento. |

### Esempi dal file
- **1X2**: probabilità 50/8.33/41.67 ⇒ quote reali 2.0/12.0/2.4. Con quote bookmaker presenti si calcola l’allibramento (6.54%) e la relativa spalmatura; la quota allibrata è il modello “sporcato” del margine.
- **OU25**: probabilità 75% sull’Over ⇒ quota reale 1.33. Quote bookmaker presenti → allibramento 5.87%.
- **OU15** e **GNG**: rispettano la stessa logica; in assenza di quote il payload mantiene `quota_reale` e lascia vuoti `quota_reale_allibrata`, `spalmatura` e `allibramento`, coerentemente con il comportamento richiesto nella codebase.

---

## 4. Blocco `metrics`

Aggrega indicatori extra calcolati in `app/analytics/metrics.py`. Attualmente contiene la sola chiave `1X2` quando il metodo “metodo_favorita” restituisce una raccomandazione; nel JSON di esempio è vuoto (`{}`) perché non è stato individuato nessun segnale.

---

## Flusso di calcolo riassunto
1. **Query DB** (`app/api/matches.py`): recupera `Match` e quote medie.
2. **Stats**: `get_team_stats` produce i conteggi; `format_stats_for_side` li incapsula nelle sezioni.
3. **Picchetto**: `calcola_picchetto_1X2` / `_binario` convertono le stats in probabilità e quote fair; se le quote bookmaker esistono calcolano l’allibramento.
4. **Metrics**: facoltativo, oggi solo sul mercato 1X2.
5. **Persistenza**: gli script di export salvano il JSON in `data/matches_api/*.json`.

Questo documento può essere riutilizzato come guida per interpretare qualsiasi file dentro `data/matches_api`.
