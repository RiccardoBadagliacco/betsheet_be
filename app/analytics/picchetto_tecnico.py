from typing import Dict, Any

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


def calcola_picchetto_structured(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcola il Picchetto Tecnico 1X2 in stile Stats4Bets a partire dalla struttura annidata.
    Restituisce:
      - quota_reale
      - quota_reale_allibrata (spalmata come fa il bookmaker)
      - allibramento_% (totale)
      - spalmatura_allibramento_% (peso % dell’allibramento)
    """
    odds = match_data.get("odds", {})
    home = match_data.get("stats_home", {}).get("stats_1_x_2", {})
    away = match_data.get("stats_away", {}).get("stats_1_x_2", {})
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
        "totali_side": (extract(home, "totali_side"), extract(away, "totali_side")),
        "recenti_side": (extract(home, "recenti_side"), extract(away, "recenti_side")),
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