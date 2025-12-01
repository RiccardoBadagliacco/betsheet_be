#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP Z — FORMATTER SMART PER IL FRONT-END
----------------------------------------------

Versione completamente riscritta:

- Prende l'output grezzo di stepZ_decision_engine (dict)
- NON ricalcola modelli: usa solo le probabilità già calcolate
- Produce un JSON compatto e leggibile per il FE, con:

    {
      "match_id": "...",
      "teams": {"home": "...", "away": "..."},
      "summary": "...",
      "direction": {...},
      "expected_goals": {...},
      "probabilities_1x2": {...},
      "multigol": {...},
      "top_scorelines": [...],
      "score_prediction": [...],
      "betting_suggestions": {...},
      "ev": {...},
      "analysis_text": "..."
    }

Pensato per essere “umano” e facilmente mappabile nel front-end.
"""

from __future__ import annotations
from typing import Dict, Any, List


# ------------------------------------------------------------
# 1) UTILS: EXPECTED GOALS, RANGES, INTENSITÀ
# ------------------------------------------------------------

def _adjust_scorelines(
    scorelines: List[Dict[str, Any]],
    p1: float,
    px: float,
    p2: float,
    o25: float | None,
    u25: float | None,
    home_yes: float | None,
    away_yes: float | None,
    cluster_1x2: int | None = None,
    cluster_ou25: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Versione C — Weighted PMF + Direction + OU + Cluster bias.
    Restituisce le scoreline ripesate e normalizzate.
    """
    if not scorelines:
        return []

    # ------------------------
    # 1) SETUP WEIGHTS
    # ------------------------
    fav_conf = max(p1, px, p2)

    # Direction bias
    if p1 == fav_conf:
        dir_boost_home = 1.25
        dir_boost_away = 0.85
    elif p2 == fav_conf:
        dir_boost_home = 0.85
        dir_boost_away = 1.25
    else:
        dir_boost_home = dir_boost_away = 1.0

    # OU bias
    if o25 is not None and o25 > 0.58:
        ou_high = 1.25   # boost punteggi con 3+ gol
        ou_low = 0.85
    elif u25 is not None and u25 > 0.58:
        ou_high = 0.85
        ou_low = 1.20    # boost punteggi 0-0, 1-1, 1-0 ecc
    else:
        ou_high = ou_low = 1.0

    # Cluster bias (solo euristico)
    cluster_factor = 1.0
    if cluster_1x2 is not None:
        if cluster_1x2 == 4:
            # cluster grandi favoriti
            cluster_factor = 1.20
        elif cluster_1x2 == 1:
            # cluster molto equilibrati
            cluster_factor = 0.95

    if cluster_ou25 is not None:
        if cluster_ou25 in (1, 2):
            # cluster ad alto punteggio
            cluster_factor *= 1.15
        elif cluster_ou25 in (4, 5):
            # cluster tendenzialmente under
            cluster_factor *= 0.90

    # ------------------------
    # 2) CALCOLO PESI SCORELINE
    # ------------------------
    weighted = []

    for s in scorelines:
        score = s["score"]
        prob = float(s["prob"])

        try:
            h, a = map(int, score.split("-"))
        except:
            h, a = 0, 0

        # direction weight
        if h > a:
            w_dir = dir_boost_home
        elif a > h:
            w_dir = dir_boost_away
        else:
            w_dir = 1.0

        # OU weight
        if (h + a) >= 3:
            w_ou = ou_high
        else:
            w_ou = ou_low

        # weight finale
        weight = prob * w_dir * w_ou * cluster_factor

        weighted.append({
            "score": score,
            "raw_prob": prob,
            "prob": weight
        })

    # ------------------------
    # 3) NORMALIZZAZIONE
    # ------------------------
    total = sum(x["prob"] for x in weighted)
    if total > 0:
        for x in weighted:
            x["prob"] = x["prob"] / total

    # ------------------------
    # 4) ORDINAMENTO
    # ------------------------
    weighted.sort(key=lambda v: v["prob"], reverse=True)

    return weighted[:5]
def _range_from_exp_goals(exp: float) -> str:
    """Mappa i gol attesi in una fascia testuale semplice."""
    if exp < 0.6:
        return "0"
    if exp < 1.3:
        return "0–1"
    if exp < 2.0:
        return "1–2"
    if exp < 3.0:
        return "2–3"
    return "3+"


def _expected_range_from_pmf(pmf: Dict[str | int, float] | None) -> str:
    """Ricava la fascia gol attesa a partire dalla pmf per squadra."""
    if not pmf:
        return "0–1"
    exp = 0.0
    for g, p in pmf.items():
        try:
            gi = int(g)
            exp += gi * float(p)
        except (TypeError, ValueError):
            continue
    return _range_from_exp_goals(exp)


def _expected_goals_from_pmf(pmf: Dict[str | int, float] | None) -> float:
    if not pmf:
        return 0.0
    exp = 0.0
    for g, p in pmf.items():
        try:
            gi = int(g)
            exp += gi * float(p)
        except (TypeError, ValueError):
            continue
    return exp


def _match_intensity_from_o25(prob: float | None) -> str:
    """Classifica l'intensità della partita dalla probabilità di Over 2.5."""
    if prob is None:
        return "medium"
    if prob < 0.42:
        return "low"
    if prob < 0.58:
        return "medium"
    return "high"


# ------------------------------------------------------------
# 2) SCORELINES & MULTIGOL
# ------------------------------------------------------------

def _top_scorelines(pmf_home: Dict[str | int, float] | None,
                    pmf_away: Dict[str | int, float] | None,
                    n: int = 5) -> List[Dict[str, Any]]:
    """
    Combina PMF home/away per generare una lista di scoreline ordinate per probabilità.
    """
    if not pmf_home or not pmf_away:
        return []

    score_probs: List[Dict[str, Any]] = []
    for gh, ph in pmf_home.items():
        for ga, pa in pmf_away.items():
            try:
                gh_i = int(gh)
                ga_i = int(ga)
                prob = float(ph) * float(pa)
            except (TypeError, ValueError):
                continue
            score_probs.append({
                "score": f"{gh_i}-{ga_i}",
                "prob": prob
            })

    score_probs.sort(key=lambda v: v["prob"], reverse=True)
    return score_probs[:n]


def _total_multigol_from_pmf(pmf_home: Dict[str | int, float] | None,
                             pmf_away: Dict[str | int, float] | None) -> Dict[str, float]:
    """
    Costruisce multigol totali di base (1-3, 1-4) usando la congiunta delle due pmf.
    """
    result = {"1-3": 0.0, "1-4": 0.0}
    if not pmf_home or not pmf_away:
        return result

    for gh, ph in pmf_home.items():
        for ga, pa in pmf_away.items():
            try:
                gh_i = int(gh)
                ga_i = int(ga)
                p = float(ph) * float(pa)
            except (TypeError, ValueError):
                continue
            tot = gh_i + ga_i
            if 1 <= tot <= 3:
                result["1-3"] += p
            if 1 <= tot <= 4:
                result["1-4"] += p

    return result


def _team_multigol_from_pmf(pmf: Dict[str | int, float] | None,
                            ranges: Dict[str, range]) -> Dict[str, float]:
    """
    Costruisce prob multigol per una singola squadra a partire dalla sua pmf.
    ranges: es: {"1-2": range(1,3), "1-3": range(1,4)}
    """
    out: Dict[str, float] = {k: 0.0 for k in ranges}
    if not pmf:
        return out

    for g, p in pmf.items():
        try:
            gi = int(g)
            prob = float(p)
        except (TypeError, ValueError):
            continue
        for label, r in ranges.items():
            if gi in r:
                out[label] += prob

    return out

# Insert _multigol_fav_from_pmf after _team_multigol_from_pmf if not present
def _multigol_fav_from_pmf(pmf: Dict[str | int, float]) -> Dict[str, float]:
    """Calcola multigol favorita da PMF naturale."""
    if not pmf:
        return {"1-2": 0.0, "1-3": 0.0, "1-4": 0.0, "1-5": 0.0, "2+": 0.0}

    ranges = {
        "1-2": range(1, 3),
        "1-3": range(1, 4),
        "1-4": range(1, 5),
        "1-5": range(1, 6),
    }

    out = {k: 0.0 for k in ranges}
    out["2+"] = 0.0

    for g, p in pmf.items():
        try:
            gi = int(g)
            pr = float(p)
        except:
            continue

        for label, r in ranges.items():
            if gi in r:
                out[label] += pr

        if gi >= 2:
            out["2+"] += pr

    return out


# ------------------------------------------------------------
# 3) SUGGERIMENTI BETTING SMART
# ------------------------------------------------------------

def _betting_suggestions_smart(
    p1: float,
    px: float,
    p2: float,
    markets: Dict[str, Any],
    ev: Dict[str, float],
    teams: Dict[str, str],
) -> Dict[str, list]:
    """
    Versione standardizzata:
    - usa "1", "X", "2"
    - usa "Casa", "Trasferta" nei multigol
    """
    sug = {"safe": [], "medium": [], "value": []}

    # Etichette standard
    home_label = "Casa"
    away_label = "Trasferta"

    # -----------------------------
    # 1) Direzione 1X2
    # -----------------------------
    fav_conf = max(p1, px, p2)
    if p2 == fav_conf:
        # favorita fuori casa
        sug["safe"].append("X2")
        sug["medium"].append("2")
    elif p1 == fav_conf:
        # favorita casa
        sug["safe"].append("1X")
        sug["medium"].append("1")
    else:
        sug["safe"].append("X")

    # -----------------------------
    # 2) Under / Over
    # -----------------------------
    o25 = markets.get("o25")
    u25 = markets.get("u25")

    if u25 and u25 > 0.55:
        if "Under 3.5" not in sug["safe"]:
            sug["safe"].append("Under 3.5")
        if "Under 2.5" not in sug["medium"]:
            sug["medium"].append("Under 2.5")

    elif o25 and o25 > 0.58:
        if "Over 2.5" not in sug["medium"]:
            sug["medium"].append("Over 2.5")

    # -----------------------------
    # 3) GG / NG
    # -----------------------------
    gg = markets.get("gg")
    if gg:
        if gg > 0.60:
            sug["medium"].append("GG")
        elif gg < 0.45:
            sug["medium"].append("NG")

    # -----------------------------
    # 4) Multigol squadra (standardizzato)
    # -----------------------------
    home_yes = markets.get("home_yes")
    away_yes = markets.get("away_yes")

    # Casa 1–3
    if home_yes and home_yes > 0.70 and p1 >= max(px, p2):
        sug["medium"].append("Multigol Casa 1–3")

    # Trasferta 1–3
    if away_yes and away_yes > 0.70 and p2 >= max(p1, px):
        sug["medium"].append("Multigol Trasferta 1–3")

    # -----------------------------
    # 5) Value bet standardizzato
    # -----------------------------
    for side, val in ev.items():
        try:
            v = float(val)
        except:
            continue
        if v > 0.07:
            side_code = {"home": "1", "draw": "X", "away": "2"}.get(side, side.upper())
            sug["value"].append(f"{side_code} (EV+{v:.2f})")

    return sug
# ------------------------------------------------------------
# 4) SUMMARY & ANALYSIS TEXT
# ------------------------------------------------------------

def _build_summary(
    teams: Dict[str, str],
    p1: float,
    px: float,
    p2: float,
    match_intensity: str,
    o25: float | None,
    u25: float | None,
) -> str:
    home = teams.get("home", "Home")
    away = teams.get("away", "Away")

    fav_conf = max(p1, px, p2)
    if p2 == fav_conf:
        fav_team = away
    elif p1 == fav_conf:
        fav_team = home
    else:
        fav_team = "pareggio"

    fav_pct = round(fav_conf * 100)
    base = f"{fav_team} favorita circa al {fav_pct}%."

    if u25 is not None and u25 > 0.55:
        goal_part = "Match tendenzialmente da under (U2.5 prevale sugli over)."
    elif o25 is not None and o25 > 0.58:
        goal_part = "Match con propensione all'over (O2.5 sopra soglia)."
    else:
        goal_part = "Equilibrio moderato sul fronte gol."

    intensity_part = {
        "low": "Ritmo atteso basso.",
        "medium": "Ritmo atteso medio.",
        "high": "Ritmo potenzialmente alto."
    }.get(match_intensity, "Ritmo atteso medio.")

    return f"{base} {goal_part} {intensity_part}"


def _build_analysis_text(
    teams: Dict[str, str],
    p1: float,
    px: float,
    p2: float,
    exp_home: float,
    exp_away: float,
    match_intensity: str,
    top_scores: List[Dict[str, Any]],
    u25: float | None,
    o25: float | None,
    ev: Dict[str, float],
) -> str:
    home = teams.get("home", "Home")
    away = teams.get("away", "Away")

    fav_conf = max(p1, px, p2)
    if p2 == fav_conf:
        fav_team = away
        fav_desc = f"{away} è leggermente favorita"
    elif p1 == fav_conf:
        fav_team = home
        fav_desc = f"{home} è leggermente favorita"
    else:
        fav_team = "pareggio"
        fav_desc = "Il pareggio è l'esito più probabile"

    fav_pct = round(fav_conf * 100)

    # parte gol
    goals_desc = f"I gol attesi sono circa {exp_home:.2f} per {home} e {exp_away:.2f} per {away}."
    if u25 is not None and u25 > 0.55:
        ou_desc = "Il modello vede una prevalenza di scenari da under 2.5."
    elif o25 is not None and o25 > 0.58:
        ou_desc = "Il modello vede una buona probabilità di over 2.5."
    else:
        ou_desc = "Il match è piuttosto neutro sul fronte over/under."

    # scoreline principali
    main_scores = ", ".join(s["score"] for s in top_scores[:3]) if top_scores else "nessuno scenario dominante"

    # value
    value_bits = []
    for side, val in ev.items():
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v > 0.07:
            side_code = {"home": "1", "draw": "X", "away": "2"}.get(side, side.upper())
            value_bits.append(f"{side_code} (EV+{v:.2f})")
    value_text = ""
    if value_bits:
        value_text = f" In chiave di value bet spiccano: {', '.join(value_bits)}."

    return (
        f"{fav_desc} (~{fav_pct}%). "
        f"{goals_desc} {ou_desc} "
        f"I risultati più coerenti con il modello sono: {main_scores}.{value_text}"
    )

def _over_under_from_pmf(pmf_home, pmf_away):
    """
    Calcola la distribuzione totale dei gol e restituisce le probabilità
    degli over/under tecnici:
      O05, O15, O25, O35, O45
      U05, U15, U25, U35, U45
    """
    if not pmf_home or not pmf_away:
        return {}

    # PMF totale gol
    pmf_tot = {}
    for gh, ph in pmf_home.items():
        for ga, pa in pmf_away.items():
            try:
                tot = int(gh) + int(ga)
                prob = float(ph) * float(pa)
            except:
                continue
            pmf_tot[tot] = pmf_tot.get(tot, 0.0) + prob

    # Helpers
    def p_under(k):
        return sum(v for g, v in pmf_tot.items() if g <= k)

    def p_over(k):
        return 1.0 - p_under(k)

    return {
        # UNDER
        "u05": p_under(0),
        "u15": p_under(1),
        "u25": p_under(2),
        "u35": p_under(3),
        "u45": p_under(4),
        # OVER
        "o05": p_over(0),
        "o15": p_over(1),
        "o25": p_over(2),
        "o35": p_over(3),
        "o45": p_over(4),
    }

import pandas as pd
from pathlib import Path

# Percorso dataset step1c (stesso usato in stepZ_decision_engine)
STEP1C_PATH = Path(__file__).resolve().parents[1] / "data" / "step1c_dataset_with_elo_form.parquet"


def _extract_recent_form(team: str, limit_date, limit_matches=10):
    """
    Restituisce le ultime N partite di una squadra con indicatori V/P/S.
    """
    try:
        df = pd.read_parquet(STEP1C_PATH)
    except Exception:
        return {"matches": [], "form_string": "", "stats": {}}

    # filtra solo prima della data del match
    df = df[df["date"] < limit_date]

    # filtra per squadra
    df_team = df[(df["home_team"] == team) | (df["away_team"] == team)]
    df_team = df_team.sort_values("date", ascending=False).head(limit_matches)

    output_matches = []
    form_seq = []

    pts = 0
    gf = 0
    ga = 0

    for _, row in df_team.iterrows():
        is_home = (row["home_team"] == team)
        goals_for = row["home_ft"] if is_home else row["away_ft"]
        goals_against = row["away_ft"] if is_home else row["home_ft"]

        # determina V/P/S
        if goals_for > goals_against:
            res = "V"
            pts += 3
        elif goals_for < goals_against:
            res = "S"
        else:
            res = "P"
            pts += 1

        gf += goals_for
        ga += goals_against

        output_matches.append({
            "date": str(row["date"]),
            "team_home": row["home_team"],
            "team_away": row["away_team"],
            # SCORE REALE DEL MATCH (non dal punto di vista della squadra)
            "score": f"{(row['home_ft'])}-{(row['away_ft'])}",
            "result": res
        })
        form_seq.append(res)

    n = len(df_team)
    stats = {
        "matches": n,
        "points": pts,
        "points_avg": round(pts / n, 2) if n > 0 else 0,
        "gf_avg": round(gf / n, 2) if n > 0 else 0,
        "ga_avg": round(ga / n, 2) if n > 0 else 0,
    }

    return {
        "matches": output_matches,
        "form_string": " ".join(form_seq),
        "stats": stats,
    }
# ------------------------------------------------------------
# 5) CORE FORMATTER
# ------------------------------------------------------------

def build_final_forecast(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    raw = output completo dello stepZ_decision_engine.

    Ritorna un dizionario pronto per FE con chiavi ad alto livello:
      - direction
      - expected_goals
      - probabilities_1x2
      - multigol
      - top_scorelines / score_prediction
      - betting_suggestions
      - ev
      - summary
      - analysis_text
    """
    teams = raw.get("teams", {}) or {}
    # -----------------------------------------
    # RECENT FORM (fallback multiplo)
    # -----------------------------------------
    match_date = (
        raw.get("match_date")
        or raw.get("meta", {}).get("match_date")
        or raw.get("affini_post", {}).get("meta", {}).get("match_date")
    )

    # Fallback finale via dataset step1c
    if match_date is None:
        try:
            df_step1c = pd.read_parquet(STEP1C_PATH)
            row = df_step1c.loc[df_step1c["match_id"] == raw.get("match_id")]
            if not row.empty:
                match_date = row.iloc[0]["date"]
        except Exception:
            pass

    recent_form = {"home": None, "away": None}

    if match_date:
        try:
            recent_form = {
                "home": _extract_recent_form(teams.get("home"), match_date),
                "away": _extract_recent_form(teams.get("away"), match_date)
            }
        except Exception:
            pass
    meta = raw.get("meta_1x2", {}) or {}
    fused = raw.get("fused_1x2", {}) or {}
    ev = raw.get("ev", {}) or {}

    aff_post = raw.get("affini_post", {}) or {}
    markets = aff_post.get("markets", {}) or {}

    # Probabilità 1X2 finali (preferisci fused, fallback meta)
    p1 = float(fused.get("p_home") or meta.get("p_home") or 0.0)
    px = float(fused.get("p_draw") or meta.get("p_draw") or 0.0)
    p2 = float(fused.get("p_away") or meta.get("p_away") or 0.0)

    # PMF home/away (da affini_post.markets.pmf)
    pmf = markets.get("pmf", {}) or {}
    pmf_home = pmf.get("home") or {}
    pmf_away = pmf.get("away") or {}

    # Scoreline intelligenti (top 5)
    top_scores = _top_scorelines(pmf_home, pmf_away, n=5)

    score_team = markets.get("score_team", {}) if isinstance(markets.get("score_team"), dict) else {}
    home_yes = score_team.get("home_yes") if score_team else markets.get("home_yes")
    away_yes = score_team.get("away_yes") if score_team else markets.get("away_yes")

    over_under_block = _over_under_from_pmf(pmf_home, pmf_away)

    # ----------------------------------------------------
    # OU 2.5 → devono essere calcolati PRIMA dello scoring
    # ----------------------------------------------------
    o25 = None
    u25 = None

    aff_1x2 = raw.get("affini_1x2", {}) or {}
    if "pO25" in aff_1x2:
        o25 = float(aff_1x2["pO25"])
    if "pU25" in aff_1x2:
        u25 = float(aff_1x2["pU25"])

    ou_market = markets.get("ou", {}) or {}
    if o25 is None and "o25" in ou_market:
        o25 = float(ou_market["o25"])
    if u25 is None and "u25" in ou_market:
        u25 = float(ou_market["u25"])

    # ORA possiamo fare l'adjust
    top_scores = _adjust_scorelines(
        top_scores,
        p1,
        px,
        p2,
        o25,
        u25,
        home_yes,
        away_yes,
        raw.get("clusters", {}).get("cluster_1x2"),
        raw.get("clusters", {}).get("cluster_ou25"),
    )


    # Expected goals
    exp_home = _expected_goals_from_pmf(pmf_home)
    exp_away = _expected_goals_from_pmf(pmf_away)

    home_range = _expected_range_from_pmf(pmf_home)
    away_range = _expected_range_from_pmf(pmf_away)

    # Intensità e OU
    o25 = None
    u25 = None
    # se presenti nella struttura affini_1x2 al livello superiore, usiamoli
    aff_1x2 = raw.get("affini_1x2", {}) or {}
    if "pO25" in aff_1x2:
        o25 = float(aff_1x2["pO25"])
    if "pU25" in aff_1x2:
        u25 = float(aff_1x2["pU25"])
    # fallback a markets.ou
    ou_market = markets.get("ou", {}) or {}
    if o25 is None and "o25" in ou_market:
        o25 = float(ou_market["o25"])
    if u25 is None and "u25" in ou_market:
        u25 = float(ou_market["u25"])

    match_intensity = _match_intensity_from_o25(o25)

    # Multigol: totali + squadra + favorita
    multigol_total = _total_multigol_from_pmf(pmf_home, pmf_away)
    multigol_home = _team_multigol_from_pmf(pmf_home, {"1-2": range(1, 3)})
    multigol_away = _team_multigol_from_pmf(pmf_away, {"1-3": range(1, 4)})

    # Determine favourite side and pick proper PMF
    if p1 >= max(px, p2):
        fav_side = "home"
        pmf_fav = pmf_home
    else:
        fav_side = "away"
        pmf_fav = pmf_away

    multigol_fav = _multigol_fav_from_pmf(pmf_fav)

    multigol_block = {
        "total": multigol_total,
        "home": multigol_home,
        "away": multigol_away,
        "fav": multigol_fav,
    }

    # Direzione finale
    fav_conf = max(p1, px, p2)
    if p2 == fav_conf:
        fav_team = teams.get("away")
        recommended = "X2"
    elif p1 == fav_conf:
        fav_team = teams.get("home")
        recommended = "1X"
    else:
        fav_team = "Draw"
        recommended = "X"

    # Suggerimenti intelligenti
    smart_suggestions = _betting_suggestions_smart(
        p1,
        px,
        p2,
        {
            "o25": o25,
            "u25": u25,
            "gg": markets.get("gg_ng", {}).get("gg") if isinstance(markets.get("gg_ng"), dict) else markets.get("gg"),
            "home_yes": markets.get("score_team", {}).get("home_yes")
            if isinstance(markets.get("score_team"), dict) else markets.get("home_yes"),
            "away_yes": markets.get("score_team", {}).get("away_yes")
            if isinstance(markets.get("score_team"), dict) else markets.get("away_yes"),
        },
        ev,
        teams,
    )

    # Summary + analysis_text
    summary = _build_summary(teams, p1, px, p2, match_intensity, o25, u25)
    analysis_text = _build_analysis_text(
        teams,
        p1,
        px,
        p2,
        exp_home,
        exp_away,
        match_intensity,
        top_scores,
        u25,
        o25,
        ev,
    )

   
    # --------------------------------------------------
    # BUILD JSON FINALE
    # --------------------------------------------------
    final = {
        "match_id": raw.get("match_id"),
        "teams": teams,
        "summary": summary,
        "direction": {
            "fav_team": fav_team,
            "confidence": round(fav_conf, 3),
            "recommended_side": recommended,
        },
        "expected_goals": {
            "home_range": home_range,
            "away_range": away_range,
            "match_intensity": match_intensity,
            "exp_home": round(exp_home, 3),
            "exp_away": round(exp_away, 3),
        },
        "probabilities_1x2": {
            "home": p1,
            "draw": px,
            "away": p2,
        },
        "multigol": multigol_block,
        # "top_scorelines": top_scores,  # Removed as requested
        "score_prediction": top_scores,
        "betting_suggestions": smart_suggestions,
        "ev": ev,
        "analysis_text": analysis_text,
        "over_under": over_under_block,
        "recent_form": recent_form,
    }

    return final


# ------------------------------------------------------------
# 6) TEST VELOCE
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Questo modulo va importato da stepZ_decision_engine, non eseguito direttamente.")