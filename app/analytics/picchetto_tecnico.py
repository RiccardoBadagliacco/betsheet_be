from typing import Dict, Any, List, Tuple, Optional


def _to_float_stat(value: Any) -> float:
    """Converte in float gestendo sia valori grezzi che dizionari {value: x}."""
    if isinstance(value, dict):
        try:
            return float(value.get("value") or 0)
        except Exception:
            return 0.0
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def _clean_odd(value: Any) -> Optional[float]:
    """Cast sicuro per quote bookmaker."""
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _calcola_picchetto_binario(
    match_data: Dict[str, Any],
    market_key: str,
    outcome_labels: Tuple[str, str],
    stat_keys: Tuple[str, str],
    odds_lookup: Dict[str, Any],
) -> Dict[str, Any]:
    """Logica condivisa per mercati binari (OU, GG/NG, ecc.)."""
    stats_home = (match_data.get("stats_home") or {}).get(market_key, {})
    stats_away = (match_data.get("stats_away") or {}).get(market_key, {})
    if not stats_home or not stats_away:
        return {}

    def read(block: Dict[str, Any], key: str) -> Tuple[float, float, float]:
        stats = block.get(key, {}).get("stats", {})
        matches = _to_float_stat(stats.get("partite"))
        first = _to_float_stat(stats.get(stat_keys[0]))
        second = _to_float_stat(stats.get(stat_keys[1]))
        return first, second, matches

    contexts: List[Tuple[str, str]] = [
        ("totali", "totali"),
        ("recenti", "recenti"),
        ("totali_home", "totali_away"),
        ("recenti_home", "recenti_away"),
    ]

    first_values: List[float] = []
    second_values: List[float] = []
    for home_key, away_key in contexts:
        h_first, h_second, h_m = read(stats_home, home_key)
        a_first, a_second, a_m = read(stats_away, away_key)
        denom = h_m + a_m
        if denom > 0:
            first_values.append((h_first + a_first) / denom)
            second_values.append((h_second + a_second) / denom)

    if not first_values or not second_values:
        return {}

    prob_first = sum(first_values) / len(first_values)
    prob_second = sum(second_values) / len(second_values)
    total_prob = prob_first + prob_second
    if total_prob == 0:
        return {}

    prob_first /= total_prob
    prob_second /= total_prob

    probs_pct = {
        outcome_labels[0]: round(prob_first * 100, 2),
        outcome_labels[1]: round(prob_second * 100, 2),
    }
    quote_reale = {
        outcome_labels[0]: round(1 / prob_first, 2) if prob_first else None,
        outcome_labels[1]: round(1 / prob_second, 2) if prob_second else None,
    }

    odds = {label: _clean_odd(odds_lookup.get(label)) for label in outcome_labels}
    has_odds = any(v and v > 0 for v in odds.values())

    quota_reale_allibrata = None
    spalmatura: Dict[str, float] = {}
    allibramento: Optional[float] = None
    if has_odds:
        prob_implicite = {k: (100 / v) for k, v in odds.items() if v and v > 0}
        somma_implicite = sum(prob_implicite.values())
        allibramento = round(somma_implicite - 100, 2) if somma_implicite else 0.0
        for segno, p in prob_implicite.items():
            peso = (p / somma_implicite * 100) if somma_implicite else 0
            spalmatura[segno] = round(peso, 2)
        quota_reale_allibrata = allibra_quote_per_spalmatura(
            quote_reali=quote_reale,
            allibramento_perc=allibramento,
            spalmatura_perc=spalmatura,
        )

    return {
        "probabilità_%": probs_pct,
        "quota_reale": quote_reale,
        "quota_reale_allibrata": quota_reale_allibrata,
        "quota_bookmaker": odds,
        "spalmatura_allibramento%": spalmatura,
        "allibramento%": allibramento,
    }

def allibra_quote_per_spalmatura(quote_reali: dict, allibramento_perc: float, spalmatura_perc: dict) -> dict:
    """
    Applica l’allibramento per segno sulla quota reale, proporzionalmente
    alla spalmatura dell’allibramento calcolata dal bookmaker.
    """
    result = {}
    over = allibramento_perc / 100.0
    for segno, q in quote_reali.items():
        if q is None:
            result[segno] = None
            continue
        peso = (spalmatura_perc.get(segno, 0) or 0) / 100.0
        margine = over * peso
        q_allibrata = q / (1 + margine)
        result[segno] = round(q_allibrata, 2)
    return result


def calcola_picchetto_1X2(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcola il Picchetto Tecnico 1X2 in stile Stats4Bets a partire dalla struttura annidata.
    Restituisce:
      - quota_reale
      - quota_reale_allibrata (spalmata come fa il bookmaker)
      - allibramento_% (totale)
      - spalmatura_allibramento_% (peso % dell’allibramento)
    """
    odds = match_data.get("odds") or {}
    home = (match_data.get("stats_home") or {}).get("1X2", {})
    away = (match_data.get("stats_away") or {}).get("1X2", {})
    home_name = match_data.get("home_name", "Casa")
    away_name = match_data.get("away_name", "Trasferta")

    # --- Helper per estrarre i valori ---
    def extract(block, key):
        b = block.get(key, {}).get("stats", {})

        def to_num(x):
            if isinstance(x, dict):
                if "value" in x:
                    try:
                        return float(x["value"])
                    except Exception:
                        return 0.0
                return 0.0
            try:
                return float(x)
            except Exception:
                return 0.0

        return to_num(b.get("vittorie", 0)), to_num(b.get("sconfitte", 0)), to_num(b.get("partite", 0))

    # --- 4 contesti (totali, recenti, totali_side, recenti_side) ---
    ctx = {
        "totali": (extract(home, "totali"), extract(away, "totali")),
        "recenti": (extract(home, "recenti"), extract(away, "recenti")),
        "totali_side": (extract(home, "totali_home"), extract(away, "totali_away")),
        "recenti_side": (extract(home, "recenti_home"), extract(away, "recenti_away")),
    }

    def pct(v1, t1, v2, t2):
        denom = t1 + t2
        return (v1 + v2) / denom if denom else 0

    # --- Calcolo delle probabilità medie (home/away) ---
    p1, p2 = [], []
    for _, ((hv, hl, ht), (av, al, at)) in ctx.items():
        p_home = pct(hv, ht, al, at)
        p_away = pct(av, at, hl, ht)
        p1.append(p_home)
        p2.append(p_away)

    prob1 = sum(p1) / len(p1)
    prob2 = sum(p2) / len(p2)
    probX = max(0, 1 - (prob1 + prob2))

    tot = prob1 + probX + prob2
    probs = {"1": prob1 / tot * 100, "X": probX / tot * 100, "2": prob2 / tot * 100}

    # --- Quote reali (senza allibramento) ---
    quote_reale = {k: round(100 / v, 2) if v > 0 else None for k, v in probs.items()}

    # --- Probabilità implicite del bookmaker ---
    prob_implicite = {k: 100 / v for k, v in odds.items() if v and v > 0}
    somma_implicite = sum(prob_implicite.values())
    allibramento = round(somma_implicite - 100, 2) if somma_implicite else 0

    # --- Spalmatura dell’allibramento (replica bookmaker) ---
    spalmatura = {}
    for segno, p in prob_implicite.items():
        peso = (p / somma_implicite * 100) if somma_implicite else 0
        spalmatura[segno] = round(peso, 2)

    # --- Applica allibramento spalmato ---
    quota_reale_allibrata = allibra_quote_per_spalmatura(
        quote_reali=quote_reale,
        allibramento_perc=allibramento,
        spalmatura_perc=spalmatura
    )

    # --- Output strutturato ---
    picchetto = {
        "probabilità_%": {k: round(v, 2) for k, v in probs.items()},
        "quota_reale": quote_reale,
        "quota_reale_allibrata": quota_reale_allibrata,
        "quota_bookmaker": odds,
        "spalmatura_allibramento%": spalmatura,
        "allibramento%": allibramento,
    }

    picchetto["analisi"] = genera_commento_picchetto(picchetto, home_name, away_name)

    return picchetto


def calcola_picchetto_ou25_structured(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calcola il Picchetto Tecnico per l'Over/Under 2.5."""
    picchetto = _calcola_picchetto_binario(
        match_data=match_data,
        market_key="OU25",
        outcome_labels=("U2.5", "O2.5"),
        stat_keys=("under", "over"),
        odds_lookup=match_data.get("odds_ou", {}) or {},
    )
    if not picchetto:
        return {}

    home_name = match_data.get("home_name", "Casa")
    away_name = match_data.get("away_name", "Trasferta")
    picchetto["analisi"] = genera_commento_picchetto_ou(picchetto, home_name, away_name, linea="2.5")
    return picchetto


def calcola_picchetto_ou15_structured(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """Picchetto Tecnico per la linea Over/Under 1.5."""
    picchetto = _calcola_picchetto_binario(
        match_data=match_data,
        market_key="OU15",
        outcome_labels=("U1.5", "O1.5"),
        stat_keys=("under", "over"),
        odds_lookup=match_data.get("odds_ou", {}) or {},
    )
    if not picchetto:
        return {}

    home_name = match_data.get("home_name", "Casa")
    away_name = match_data.get("away_name", "Trasferta")
    picchetto["analisi"] = genera_commento_picchetto_ou(picchetto, home_name, away_name, linea="1.5")
    return picchetto


def calcola_picchetto_gng_structured(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """Picchetto Tecnico per il mercato Goal/NoGoal."""
    picchetto = _calcola_picchetto_binario(
        match_data=match_data,
        market_key="GNG",
        outcome_labels=("GG", "NG"),
        stat_keys=("goal", "no_goal"),
        odds_lookup=match_data.get("odds_gng", {}) or {},
    )
    if not picchetto:
        return {}

    home_name = match_data.get("home_name", "Casa")
    away_name = match_data.get("away_name", "Trasferta")
    picchetto["analisi"] = genera_commento_picchetto_gng(picchetto, home_name, away_name)
    return picchetto


# ----------------------------------------------
# COMMENTO INTELLIGENTE
# ----------------------------------------------

def genera_commento_picchetto(picchetto: dict, home_name: str, away_name: str) -> str:
    """
    Genera un commento sintetico ma qualitativo (4-5 righe),
    analizzando il picchetto tecnico completo e concludendo
    con una valutazione chiara e breve.
    """
    probs = picchetto.get("probabilità_%", {})
    q_allib = picchetto.get("quota_reale_allibrata", {})
    q_book = picchetto.get("quota_bookmaker", {})
    allib = picchetto.get("allibramento%", 0)
    spalm = picchetto.get("spalmatura_allibramento%", {})

    if not probs or not q_book:
        return "⚠️ Dati insufficienti per generare l'analisi."

    differenze = {}
    for segno in ["1", "X", "2"]:
        qr = q_allib.get(segno)
        qb = q_book.get(segno)
        if qr and qb:
            differenze[segno] = ((qb - qr) / qr) * 100

    favorito = max(probs, key=probs.get)
    secondo = sorted(probs.items(), key=lambda x: x[1], reverse=True)[1][0]
    sottostimati = [s for s, d in differenze.items() if d > 10]
    sovrastimati = [s for s, d in differenze.items() if d < -10]

    testo = []

    # 1️⃣ Lettura generale
    if favorito == "1":
        testo.append(f"📊 Il modello favorisce **{home_name}** ({probs['1']:.1f}%) rispetto a **{away_name}** ({probs['2']:.1f}%).")
    elif favorito == "2":
        testo.append(f"📊 Il modello individua **{away_name}** come favorito ({probs['2']:.1f}%), ma lascia margine al {home_name} ({probs['1']:.1f}%).")
    else:
        testo.append("📊 Match equilibrato, nessuna squadra emerge chiaramente.")

    # 2️⃣ Bookmaker
    segno_pressato = max(spalm, key=spalm.get)
    testo.append(f"💹 Allibramento del **{allib:.2f}%**, con maggiore pressione su **{segno_pressato}** ({spalm[segno_pressato]:.1f}%).")

    # 3️⃣ Value e sintesi qualitativa
    if sottostimati:
        testo.append(f"💰 Il modello trova **value** sui segni **{', '.join(sottostimati)}**, sottovalutati dal mercato.")
    elif sovrastimati:
        testo.append(f"⚠️ I segni **{', '.join(sovrastimati)}** appaiono **sopravvalutati** dal bookmaker.")
    else:
        testo.append("ℹ️ Nessuna divergenza significativa tra modello e mercato.")

    # 4️⃣ Conclusione sintetica
    if sottostimati:
        testo.append(f"👉 **Conclusione:** possibile valore su {', '.join(sottostimati)}.")
    elif sovrastimati:
        testo.append(f"👉 **Conclusione:** mercato sbilanciato su {', '.join(sovrastimati)}.")
    else:
        testo.append("👉 **Conclusione:** equilibrio statistico, nessun valore chiaro.")

    return "\n".join(testo)


def genera_commento_picchetto_ou(picchetto: dict, home_name: str, away_name: str, linea: str = "2.5") -> str:
    probs = picchetto.get("probabilità_%", {})
    allib = picchetto.get("allibramento%", 0)
    spalm = picchetto.get("spalmatura_allibramento%", {})
    q_allib = picchetto.get("quota_reale_allibrata", {})
    q_book = picchetto.get("quota_bookmaker", {})

    under = probs.get("U2.5")
    over = probs.get("O2.5")
    if under is None or over is None:
        return "⚠️ Dati insufficienti per la linea 2.5 gol."

    testo = []
    if under > over + 5:
        testo.append(f"📊 Modello orientato a un match chiuso: Under 2.5 {under:.1f}%.")
    elif over > under + 5:
        testo.append(f"📊 Modello vede gara aperta: Over 2.5 {over:.1f}%.")
    else:
        testo.append("📊 Equilibrio statistico sulla linea 2.5 gol.")

    if q_allib and q_book:
        value_under = None
        if q_allib.get("U2.5") and q_book.get("U2.5"):
            value_under = ((q_book["U2.5"] - q_allib["U2.5"]) / q_allib["U2.5"]) * 100
        value_over = None
        if q_allib.get("O2.5") and q_book.get("O2.5"):
            value_over = ((q_book["O2.5"] - q_allib["O2.5"]) / q_allib["O2.5"]) * 100

        value_msgs = []
        if value_under is not None and value_under > 10:
            value_msgs.append("Under 2.5")
        if value_over is not None and value_over > 10:
            value_msgs.append("Over 2.5")
        if value_msgs:
            testo.append(f"💰 Possibile value su {', '.join(value_msgs)} rispetto al mercato.")

    if spalm:
        segno_pressato = max(spalm, key=spalm.get)
        testo.append(
            f"💹 Allibramento {allib:.2f}% con maggior pressione su {segno_pressato} ({spalm[segno_pressato]:.1f}%)."
        )

    testo.append(f"👉 Focus match: {home_name} vs {away_name} sulla linea {linea} gol.")
    return "\n".join(testo)


def genera_commento_picchetto_gng(picchetto: dict, home_name: str, away_name: str) -> str:
    probs = picchetto.get("probabilità_%", {})
    gg = probs.get("GG")
    ng = probs.get("NG")
    if gg is None or ng is None:
        return "⚠️ Dati insufficienti per il mercato Goal/No Goal."

    testo = []
    if gg > ng + 5:
        testo.append(f"📊 Modello orientato al GG ({gg:.1f}%), entrambe le squadre possono andare a segno.")
    elif ng > gg + 5:
        testo.append(f"📊 Tendenza verso il NoGol ({ng:.1f}%), match che può rimanere bloccato.")
    else:
        testo.append("📊 Equilibrio statistico sul mercato GG/NG.")

    q_allib = picchetto.get("quota_reale_allibrata")
    q_book = picchetto.get("quota_bookmaker", {})
    value_msgs = []
    if q_allib and q_book:
        if q_allib.get("GG") and q_book.get("GG"):
            value = ((q_book["GG"] - q_allib["GG"]) / q_allib["GG"]) * 100
            if value > 10:
                value_msgs.append("GG")
        if q_allib.get("NG") and q_book.get("NG"):
            value = ((q_book["NG"] - q_allib["NG"]) / q_allib["NG"]) * 100
            if value > 10:
                value_msgs.append("NG")
    if value_msgs:
        testo.append(f"💰 Possibile value sui segni {', '.join(value_msgs)} rispetto al bookmaker.")

    spalm = picchetto.get("spalmatura_allibramento%", {})
    allib = picchetto.get("allibramento%")
    if spalm:
        segno_pressato = max(spalm, key=spalm.get)
        testo.append(
            f"💹 Allibramento {allib:.2f}% con maggior esposizione su {segno_pressato} ({spalm[segno_pressato]:.1f}%)."
        )

    testo.append(f"👉 Focus match: {home_name} vs {away_name} per GG/NoGol.")
    return "\n".join(testo)
