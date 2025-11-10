from typing import Dict, Any
from statistics import mean


# ---------------------------- FUNZIONI DI SUPPORTO ---------------------------- #

def _safe_get(d: Dict[str, Any], *keys, default=0.0) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            return float(d[k])
    return default


def normalizza_probabilita(prob_dict: Dict[str, float]) -> Dict[str, float]:
    s = sum(prob_dict.values())
    if s == 0:
        return {k: 0.0 for k in prob_dict}
    factor = 100.0 / s
    return {k: v * factor for k, v in prob_dict.items()}


def prob_from_quote(quote: float) -> float:
    return round((1 / quote) * 100, 2) if quote > 0 else 0.0


def applica_allibramento(quote_reali_pure: dict, odds_book: dict) -> tuple[dict, float]:
    """Applica l'overround del bookmaker alle quote reali (no rinormalizzazione)."""
    S = sum((1 / q) for q in odds_book.values() if q and q > 0)
    P_star = {k: 1 / q for k, q in quote_reali_pure.items() if q and q > 0}
    P_all = {k: v * S for k, v in P_star.items()}
    Q_all = {k: round(1 / v, 2) for k, v in P_all.items()}
    overround = round((S - 1) * 100, 2)
    return Q_all, overround


# ---------------------------- CALCOLO PRINCIPALE ---------------------------- #

def calcola_picchetto(match: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcola tutte le informazioni principali del Picchetto Tecnico Stats4Bets.
    Restituisce un dizionario strutturato, pronto per l’uso API o analitico.
    """
    odds = match.get("odds", {})
    sh = match.get("stats_home", {})
    sa = match.get("stats_away", {})

    # Calcolo probabilità base per segno 1 e 2
    p1 = mean([
        (_safe_get(sh, 'vittorie_totali') + _safe_get(sa, 'sconfitte_totali')) /
        (_safe_get(sh, 'partite_totali') + _safe_get(sa, 'partite_totali')) * 100,
        (_safe_get(sh, 'vittorie_recenti') + _safe_get(sa, 'sconfitte_recenti')) /
        (_safe_get(sh, 'partite_recenti') + _safe_get(sa, 'partite_recenti')) * 100,
        (_safe_get(sh, 'vittorie_casa') + _safe_get(sa, 'sconfitte_trasferta', 'sconfitte_totali')) /
        (_safe_get(sh, 'partite_casa') + _safe_get(sa, 'partite_trasferta', 'partite_totali')) * 100,
        (_safe_get(sh, 'vittorie_casa_recenti') + _safe_get(sa, 'sconfitte_trasferta_recenti', 'sconfitte_recenti')) /
        (_safe_get(sh, 'partite_casa_recenti') + _safe_get(sa, 'partite_trasferta_recenti', 'partite_recenti')) * 100
    ])

    p2 = mean([
        (_safe_get(sa, 'vittorie_totali') + _safe_get(sh, 'sconfitte_totali')) /
        (_safe_get(sa, 'partite_totali') + _safe_get(sh, 'partite_totali')) * 100,
        (_safe_get(sa, 'vittorie_recenti') + _safe_get(sh, 'sconfitte_recenti')) /
        (_safe_get(sa, 'partite_recenti') + _safe_get(sh, 'partite_recenti')) * 100,
        (_safe_get(sa, 'vittorie_trasferta', 'vittorie_totali') + _safe_get(sh, 'sconfitte_casa')) /
        (_safe_get(sa, 'partite_trasferta', 'partite_totali') + _safe_get(sh, 'partite_casa')) * 100,
        (_safe_get(sa, 'vittorie_trasferta_recenti', 'vittorie_recenti') + _safe_get(sh, 'sconfitte_casa_recenti')) /
        (_safe_get(sa, 'partite_trasferta_recenti', 'partite_recenti') + _safe_get(sh, 'partite_casa_recenti')) * 100
    ])

    # Pareggio per compensazione
    pX = 100 - (p1 + p2)

    # Normalizza le probabilità
    prob_tecniche = normalizza_probabilita({"1": p1, "X": pX, "2": p2})

    # Quote reali pure
    quote_reali_pure = {k: round(100.0 / v, 2) for k, v in prob_tecniche.items()}

    # Quote reali allibrate
    quote_reali_allibrate, allibramento = applica_allibramento(quote_reali_pure, odds)

    # Value % (reale allibrata - book) / book * 100
    value_diff = {
        k: round(((quote_reali_allibrate[k] - odds[k]) / odds[k]) * 100, 2)
        for k in ["1", "X", "2"]
        if odds.get(k) and quote_reali_allibrate.get(k)
    }

    # Probabilità implicite bookmaker
    prob_book = {k: prob_from_quote(v) for k, v in odds.items()}

    # Interpretazione in stile Stats4Bets
    interpretazioni = {}
    for segno in ["1", "X", "2"]:
        val = value_diff.get(segno, 0)
        if val < -5:
            interpretazioni[segno] = "🟢 Value reale (book paga più del giusto)"
        elif val > 5:
            interpretazioni[segno] = "🔴 Quota strozzata (book paga meno del giusto)"
        else:
            interpretazioni[segno] = "🟡 Quota coerente"

    # Analisi di mercato (divergenza picchetto vs book)
    favorito_tecnico = max(prob_tecniche, key=prob_tecniche.get)
    favorito_book = min(odds, key=odds.get)
    alert_mercato = None
    if favorito_tecnico != favorito_book:
        alert_mercato = f"⚠️ Divergenza: Picchetto → {favorito_tecnico}, Book → {favorito_book}"
    else:
        alert_mercato = f"✅ Coerenza: entrambi favoriscono {favorito_book}"

    # Struttura dati finale
    return {
        "allibramento_%": allibramento,
        "prob_tecniche": prob_tecniche,
        "quote_reali_pure": quote_reali_pure,
        "quote_reali_allibrate": quote_reali_allibrate,
        "prob_bookmaker": prob_book,
        "odds_bookmaker": odds,
        "value_percent": value_diff,
        "interpretazioni": interpretazioni,
        "alert_mercato": alert_mercato
    }


# ---------------------------- ESEMPIO ---------------------------- #

if __name__ == "__main__":
    match = {
        "odds": {"1": 1.75, "X": 3.60, "2": 4.75},
        "stats_home": {
            "vittorie_totali": 6, "sconfitte_totali": 9, "partite_totali": 20,
            "vittorie_recenti": 2, "sconfitte_recenti": 2, "partite_recenti": 5,
            "vittorie_casa": 2, "sconfitte_casa": 6, "partite_casa": 10,
            "vittorie_casa_recenti": 1, "sconfitte_casa_recenti": 2, "partite_casa_recenti": 5
        },
        "stats_away": {
            "vittorie_totali": 6, "sconfitte_totali": 7, "partite_totali": 20,
            "vittorie_recenti": 1, "sconfitte_recenti": 2, "partite_recenti": 5,
            "vittorie_trasferta": 4, "sconfitte_trasferta": 3, "partite_trasferta": 9,
            "vittorie_trasferta_recenti": 0, "sconfitte_trasferta_recenti": 3, "partite_trasferta_recenti": 5
        }
    }

    result = calcola_picchetto(match)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
