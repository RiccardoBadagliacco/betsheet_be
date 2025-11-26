# ---------------------------------------------------------
# Forbice helper
# ---------------------------------------------------------
# ---------------------------------------------------------
# Forbice helper
# ---------------------------------------------------------
def is_forbice(row) -> bool:
    """
    Metrica Forbice:
    - sia bk_q1 che bk_q2 sono nell'intervallo [2.20, 3.10]
    """
    try:
        q1 = float(row.bk_q1)
        q2 = float(row.bk_q2)
    except Exception:
        return False

    return (2.20 <= q1 <= 3.10) and (2.20 <= q2 <= 3.10)

# ---------------------------------------------------------
# MegaX helper
# ---------------------------------------------------------
# ---------------------------------------------------------
# MegaX helper
# ---------------------------------------------------------
def is_megaX(row) -> bool:
    """
    MegaX: la X è il segno con quota più bassa.
    """
    try:
        q1 = float(row.bk_q1)
        qx = float(row.bk_qx)
        q2 = float(row.bk_q2)
    except Exception:
        return False
    if any(np.isnan(x) for x in [q1, qx, q2]):
        return False
    return (qx < q1) and (qx < q2)

# ---------------------------------------------------------
# Metriche sperimentali (ispirate alla letteratura)
# ---------------------------------------------------------
def is_fibonacci_odds(row, tol_ratio_low: float = 1.4, tol_ratio_high: float = 1.8) -> bool:
    """
    Metrica "Fibonacci Odds":
    Le tre quote 1-X-2 seguono approssimativamente una progressione
    in cui i rapporti tra le quote ordinate sono vicini a ~1.6 (es. 1.6–2.6–4.2).

    Implementazione:
    - ordiniamo le tre quote in senso crescente
    - calcoliamo r1 = q2/q1, r2 = q3/q2
    - true se r1 e r2 sono entrambi in [tol_ratio_low, tol_ratio_high]
    """
    try:
        q1 = float(row.bk_q1)
        qx = float(row.bk_qx)
        q2 = float(row.bk_q2)
    except Exception:
        return False

    if any(np.isnan(x) for x in [q1, qx, q2]):
        return False

    qs = sorted([q1, qx, q2])
    if qs[0] <= 0 or qs[1] <= 0:
        return False

    r1 = qs[1] / qs[0]
    r2 = qs[2] / qs[1]

    return (tol_ratio_low <= r1 <= tol_ratio_high) and (tol_ratio_low <= r2 <= tol_ratio_high)


def is_elastic_20(row, max_spread_pct: float = 0.20) -> bool:
    """
    Metrica "Elasticità 20%":
    Le tre quote 1-X-2 sono tutte relativamente vicine tra loro
    (partita molto equilibrata).

    Implementazione:
    - calcoliamo spread = (q_max - q_min) / q_min
    - true se spread <= max_spread_pct
    """
    try:
        q1 = float(row.bk_q1)
        qx = float(row.bk_qx)
        q2 = float(row.bk_q2)
    except Exception:
        return False

    if any(np.isnan(x) for x in [q1, qx, q2]):
        return False

    q_min = min(q1, qx, q2)
    q_max = max(q1, qx, q2)
    if q_min <= 0:
        return False

    spread = (q_max - q_min) / q_min
    return spread <= max_spread_pct


def is_disorder_x_high(row) -> bool:
    """
    Metrica "Disordine del mercato":
    La X è la quota più alta (il segno meno probabile secondo il mercato).
    """
    try:
        q1 = float(row.bk_q1)
        qx = float(row.bk_qx)
        q2 = float(row.bk_q2)
    except Exception:
        return False

    if any(np.isnan(x) for x in [q1, qx, q2]):
        return False

    return (qx > q1) and (qx > q2)


def classify_double_gradient(row, g1_min: float = 0.5, g2_max: float = 0.2):
    """
    Metrica "Doppio Gradiente":

    Ordiniamo le quote in senso crescente: q0 <= q1 <= q2
    - g1 = q1 - q0
    - g2 = q2 - q1

    Se g1 >= g1_min e g2 <= g2_max  → pattern WEAK_GRADIENT
    Se g1 <= g2_max e g2 >= g1_min  → pattern STRONG_GRADIENT

    Ritorna:
      "WEAK", "STRONG" oppure None
    """
    try:
        q1 = float(row.bk_q1)
        qx = float(row.bk_qx)
        q2 = float(row.bk_q2)
    except Exception:
        return None

    if any(np.isnan(x) for x in [q1, qx, q2]):
        return None

    qs = sorted([q1, qx, q2])
    g1 = qs[1] - qs[0]
    g2 = qs[2] - qs[1]

    if (g1 >= g1_min) and (g2 <= g2_max):
        return "WEAK"
    if (g1 <= g2_max) and (g2 >= g1_min):
        return "STRONG"
    return None


def is_compression_quotes(row, max_std: float = 0.45) -> bool:
    """
    Metrica "Compressione dei segni":
    Lo scarto fra le tre quote 1-X-2 è molto ridotto
    (deviazione standard bassa).

    Implementazione:
    - std(quote) < max_std
    """
    try:
        q1 = float(row.bk_q1)
        qx = float(row.bk_qx)
        q2 = float(row.bk_q2)
    except Exception:
        return False

    if any(np.isnan(x) for x in [q1, qx, q2]):
        return False

    arr = np.array([q1, qx, q2], dtype=float)
    return float(arr.std()) < max_std

def is_book_shape(row, tolerance: float = 0.15) -> bool:
    """
    Metrica Book Shape (Triangle Bias):
    Le tre quote devono formare un profilo triangolare stabile:
    - la favorita ha quota bassa
    - la X è circa il 40-60% più alta della favorita
    - la sfavorita è circa 2.0-3.0 volte la favorita
    """
    try:
        q1 = float(row.bk_q1)
        qx = float(row.bk_qx)
        q2 = float(row.bk_q2)
    except:
        return False
    if any(np.isnan([q1,qx,q2])):
        return False
    fav = min(q1,qx,q2)
    if fav <= 0: return False
    ratio_x = qx / fav
    ratio_dog = max(q1,q2) / fav
    return (1.4 - tolerance <= ratio_x <= 1.6 + tolerance) and (2.0 - tolerance <= ratio_dog <= 3.0 + tolerance)

def is_draw_magnet(row, target_min: float = 3.20, target_max: float = 3.60) -> bool:
    """
    Draw Magnet:
    La quota X è molto stabile e centrata nel range 3.20–3.60.
    """
    try:
        qx = float(row.bk_qx)
    except:
        return False
    if np.isnan(qx): return False
    return target_min <= qx <= target_max

def is_balanced_shape(row, tol: float = 0.25) -> bool:
    """
    Balanced shape 2.50–3.20–2.50:
    Le quote 1 e 2 sono simili, la X più alta di ~0.7
    """
    try:
        q1 = float(row.bk_q1)
        qx = float(row.bk_qx)
        q2 = float(row.bk_q2)
    except:
        return False
    if any(np.isnan([q1,qx,q2])): return False
    return (abs(q1 - 2.50) <= tol and abs(q2 - 2.50) <= tol and abs(qx - 3.20) <= tol)

def is_double_favorite_compression(row, max_diff: float = 0.25) -> bool:
    """
    Double Favorite Compression (DFC):
    Le quote di 1 e 2 sono entrambe molto basse e vicine.
    """
    try:
        q1 = float(row.bk_q1)
        q2 = float(row.bk_q2)
    except:
        return False
    if any(np.isnan([q1,q2])): return False
    return (q1 < 2.20 and q2 < 2.20 and abs(q1 - q2) <= max_diff)


# ---------------------------------------------------------
# Nuove metriche sperimentali
# ---------------------------------------------------------
def is_volatility_pattern(row, threshold: float = 0.40) -> bool:
    """
    Metrica Volatilità:
    Se tra le tre quote c'è una variazione molto forte (range > 40% rispetto alla quota min),
    spesso la partita è poco prevedibile e aumenta la probabilità di Over 2.5.
    """
    try:
        q1 = float(row.bk_q1)
        qx = float(row.bk_qx)
        q2 = float(row.bk_q2)
    except:
        return False
    if any(np.isnan([q1,qx,q2])):
        return False
    qmin = min(q1,qx,q2)
    qmax = max(q1,qx,q2)
    if qmin <= 0:
        return False
    spread = (qmax - qmin) / qmin
    return spread >= threshold


def is_inverse_favorite_bias(row, tolerance: float = 0.20) -> bool:
    """
    Metrica Inverse Favorite Bias:
    La favorita del bookmaker ha una quota significativamente più ALTA della
    favorita tecnica → indicazione di possibile errore di pricing.
    """
    try:
        q1_b, qx_b, q2_b = float(row.bk_q1), float(row.bk_qx), float(row.bk_q2)
        q1_t, qx_t, q2_t = float(row.pt_q1), float(row.pt_qx), float(row.pt_q2)
    except:
        return False
    if any(np.isnan([q1_b,qx_b,q2_b,q1_t,qx_t,q2_t])):
        return False

    # favorita
    fav_b = min([q1_b,qx_b,q2_b])
    fav_t = min([q1_t,qx_t,q2_t])
    if fav_t <= 0: 
        return False

    diff = (fav_b - fav_t) / fav_t
    return diff > tolerance
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BSV – BetSheet Value System
===========================

Analizza:
- quote tecniche dal picchetto tecnico
- quote reali bookmaker

e determina per ogni match:

    NO_BVS
    SEMI_BVS
    BVS
        - BVS_lineare
        - BVS_non_lineare_1
        - BVS_non_lineare_2
        - BVS_non_lineare_DC12
        - BVS_1X
        - BVS_X2

Output: parquet con classificazione finale.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PICCHETTO_FILE = DATA_DIR / "picchetto_tecnico_1x2.parquet"
BOOK_FILE = DATA_DIR / "quote_book.parquet"  # <-- sostituisci se diverso
BOOK_FALLBACK = (
    BASE_DIR.parent / "correlazioni_affini_v2" / "data" / "step0_dataset_base.parquet"
)
OUT_FILE = DATA_DIR / "bsv_output.parquet"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def order_signs(q1, qx, q2):
    """
    Ritorna l'ordine dei segni in base alle quote:
    quota più bassa = più favorita.

    Output es:
        ['1', 'X', '2']
    """
    arr = [
        ('1', q1),
        ('X', qx),
        ('2', q2),
    ]
    arr_sorted = sorted(arr, key=lambda x: x[1])
    return [x[0] for x in arr_sorted]


# ---------------------------------------------------------
# Formula 4 helpers
# ---------------------------------------------------------
def is_formula4_home(row, min_odd: float = 3.75, max_odd: float = 5.25) -> bool:
    """
    Formula 4 (versione HOME):

    - quota 1 e quota X "gravitano" intorno a 4
      → entrambe nell'intervallo [min_odd, max_odd]

    Quando accade:
      - ci aspettiamo almeno 2 gol nel match (Over 1.5)
      - ci aspettiamo almeno 1 gol squadra di casa
    """
    try:
        q1 = float(row.bk_q1)
        qx = float(row.bk_qx)
    except Exception:
        return False

    if np.isnan(q1) or np.isnan(qx):
        return False

    return (min_odd <= q1 <= max_odd) and (min_odd <= qx <= max_odd)


def is_formula4_away(row, min_odd: float = 3.75, max_odd: float = 5.25) -> bool:
    """
    Formula 4 (versione AWAY):

    - quota X e quota 2 "gravitano" intorno a 4
      → entrambe nell'intervallo [min_odd, max_odd]

    Quando accade:
      - ci aspettiamo almeno 2 gol nel match (Over 1.5)
      - ci aspettiamo almeno 1 gol squadra ospite
    """
    try:
        q2 = float(row.bk_q2)
        qx = float(row.bk_qx)
    except Exception:
        return False

    if np.isnan(q2) or np.isnan(qx):
        return False

    return (min_odd <= q2 <= max_odd) and (min_odd <= qx <= max_odd)


def classify_bvs(row):
    """Applica tutte le regole BVS a una singola riga."""

    # Ordine bookmaker
    bk_order = order_signs(row.bk_q1, row.bk_qx, row.bk_q2)

    # Ordine picchetto tecnico
    pt_order = order_signs(row.pt_q1, row.pt_qx, row.pt_q2)

    # Estraggo favorita e seconda
    bk_fav, bk_second, _ = bk_order
    pt_fav, pt_second, _ = pt_order


    # ---------------------------------------------
    # 1) NO_BVS – SEMI_BVS – BVS
    # ---------------------------------------------
    if bk_fav == pt_fav and bk_second == pt_second:
        base = "BVS"
    elif bk_fav == pt_fav:
        base = "SEMI_BVS"
    else:
        return "NO_BVS"

    # Se SEMI_BVS → finito
    if base == "SEMI_BVS":
        return "SEMI_BVS"

    # Da qui in poi → è un vero BVS
    # ---------------------------------------------
    # 2) BVS LINEARE (raffinato "alla letteratura")
    # ---------------------------------------------
    fav_quote_bk = {
        '1': row.bk_q1,
        'X': row.bk_qx,
        '2': row.bk_q2
    }[bk_fav]

    fav_quote_pt = {
        '1': row.pt_q1,
        'X': row.pt_qx,
        '2': row.pt_q2
    }[pt_fav]

    if fav_quote_pt > 0:
        diff_perc = abs(fav_quote_bk - fav_quote_pt) / fav_quote_pt
    else:
        diff_perc = 999.0

    # Quote bookmaker in float per controlli aggiuntivi
    try:
        q1_b = float(row.bk_q1)
        qx_b = float(row.bk_qx)
        q2_b = float(row.bk_q2)
        q1_t = float(row.pt_q1)
        qx_t = float(row.pt_qx)
        q2_t = float(row.pt_q2)
    except Exception:
        q1_b = qx_b = q2_b = q1_t = qx_t = q2_t = np.nan

    # Pattern "scala" sui bookmaker e sul picchetto
    scala_bk_1x2 = (q1_b < qx_b < q2_b)
    scala_bk_2x1 = (q2_b < qx_b < q1_b)
    scala_pt_1x2 = (q1_t < qx_t < q2_t)
    scala_pt_2x1 = (q2_t < qx_t < q1_t)

    # BVS lineare solo se:
    #  - differenza favorita book vs tecnica < 20%
    #  - favorita è 1 o 2 (escludiamo X-favorite)
    #  - pattern di scala coerente fra bookmaker e picchetto
    #  - favorita del book non sopra 1.90 (range "media-bassa" della letteratura)
    if diff_perc < 0.20 and bk_fav in ('1', '2') and fav_quote_bk <= 1.90:
        scala_ok = False
        if bk_fav == '1':
            scala_ok = scala_bk_1x2 and scala_pt_1x2
        elif bk_fav == '2':
            scala_ok = scala_bk_2x1 and scala_pt_2x1

        if scala_ok:
            # Filtri qualitativi: evitiamo BVS_lineare in contesti estremi
            try:
                # 1) pareggio troppo alto  → mercato poco stabile sulla X
                if qx_b > 4.0:
                    scala_ok = False
            except Exception:
                pass

            try:
                # 2) volatilità estrema delle quote (range > 40% rispetto alla min)
                if is_volatility_pattern(row):
                    scala_ok = False
            except Exception:
                pass

        if scala_ok:
            return "BVS_lineare"
    # Se le condizioni per il lineare non passano, si prosegue con le varianti non lineari.

    # ---------------------------------------------
    # 3) BVS NON-LINEARE 1 (favorita casa)
    # ---------------------------------------------
    if bk_fav == '1' and row.bk_q1 < row.pt_q1 and row.bk_q1 < 1.65:
        return "BVS_non_lineare_1"

    # ---------------------------------------------
    # 4) BVS NON-LINEARE 2 (favorita trasferta)
    # ---------------------------------------------
    if bk_fav == '2' and row.bk_q2 < row.pt_q2 and row.bk_q2 < 1.65:
        return "BVS_non_lineare_2"

    # ---------------------------------------------
    # 5) BVS NON-LINEARE DC12 (raffinato)
    # ---------------------------------------------
    DR = abs(row.bk_q1 - row.bk_q2)
    DT = abs(row.pt_q1 - row.pt_q2)

    if DT < DR:
        # Escludiamo contesti di:
        #  - forte bias sul pareggio (X molto bassa)
        #  - match "Forbice" (1 e 2 intorno a 2.2–3.1), più adatti ad analisi over/under
        try:
            if float(row.bk_qx) < 3.30:
                return "NO_BVS"
        except Exception:
            return "NO_BVS"

        try:
            if is_forbice(row):
                return "NO_BVS"
        except Exception:
            pass

        # Inoltre richiediamo una vera asimmetria fra 1 e 2,
        # ma senza quote estreme tipo 1.10 o 7.00.
        try:
            if not (1.30 <= q1_b <= 3.80 and 1.30 <= q2_b <= 3.80):
                return "NO_BVS"
            if abs(q1_b - q2_b) < 0.30:  # troppo compressi per una DC12 "pulita"
                return "NO_BVS"
        except Exception:
            pass

        return "BVS_non_lineare_DC12"

    # ---------------------------------------------
    # 6) fallback → 1X o X2 SOLO se scala forte
    # ---------------------------------------------
    # Qui trattiamo i BVS residui come doppia chance attorno alla favorita,
    # ma SOLO se la struttura delle quote è una vera scala (1X2 o 2X1)
    # e la favorita è in range <=1.90.
    if bk_fav in ('1', '2') and fav_quote_bk <= 1.90:
        if bk_fav == '1' and scala_bk_1x2:
            return "BVS_1X"
        if bk_fav == '2' and scala_bk_2x1:
            return "BVS_X2"

    # Se nessun pattern forte è presente, non consideriamo il match come BVS utilizzabile.
    return "NO_BVS"


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def load_picchetto():
    if not PICCHETTO_FILE.exists():
        raise FileNotFoundError(f"File picchetto non trovato: {PICCHETTO_FILE}")
    print(f"📥 Carico picchetto tecnico da {PICCHETTO_FILE}")
    return pd.read_parquet(PICCHETTO_FILE), PICCHETTO_FILE


def load_bookmaker_quotes():
    """
    Carica le quote bookmaker. Se quote_book.parquet non esiste,
    prova a usare step0_dataset_base.parquet come fallback.
    """
    if BOOK_FILE.exists():
        print(f"📥 Carico quote bookmaker da {BOOK_FILE}")
        return pd.read_parquet(BOOK_FILE), BOOK_FILE

    if BOOK_FALLBACK.exists():
        print(f"⚠️  {BOOK_FILE.name} non trovato, uso fallback: {BOOK_FALLBACK.name}")
        bk = pd.read_parquet(BOOK_FALLBACK)
        # Mappa le colonne di step0_dataset_base alle attese
        rename_map = {
            "avg_home_odds": "bk_q1",
            "avg_draw_odds": "bk_qx",
            "avg_away_odds": "bk_q2",
        }
        bk = bk.rename(columns=rename_map)
        missing = [c for c in ("bk_q1", "bk_qx", "bk_q2", "match_id") if c not in bk]
        if missing:
            raise ValueError(f"Mancano colonne nel fallback: {missing}")
        return bk[["match_id", "bk_q1", "bk_qx", "bk_q2"]], BOOK_FALLBACK

    raise FileNotFoundError(
        f"File quote bookmaker non trovato. Attesi: {BOOK_FILE} oppure {BOOK_FALLBACK}"
    )


def main():
    pic, pic_path = load_picchetto()
    bk, bk_path = load_bookmaker_quotes()

    print(f"   → Picchetto righe: {len(pic):,}")
    print(f"   → Bookmaker righe: {len(bk):,}")

    # Allineo il tipo della chiave di merge
    pic["match_id"] = pic["match_id"].astype(str)
    bk["match_id"] = bk["match_id"].astype(str)

    # Merge on match_id
    df = pic.merge(bk, on="match_id", how="inner")
    print(
        f"🔗 Merge: {len(df):,} match comuni "
        f"({len(df)/len(pic)*100:.1f}% di picchetto, {len(df)/len(bk)*100:.1f}% di book)"
    )

    # Filtro: considera solo partite con picchetto tecnico valido
    before_drop = len(df)
    df = df.dropna(subset=["pt_q1", "pt_qx", "pt_q2"])
    if len(df) != before_drop:
        print(f"⚙️  Filtrate righe senza pt_q*: {before_drop - len(df):,} scartate")

    # Calcolo classificazione BVS
    print("🧠 Calcolo classificazione BSV…")
    df["BSV_type"] = df.apply(classify_bvs, axis=1)

    # compute actual sign
    def actual_sign(row):
        if row.gol_home > row.gol_away:
            return '1'
        elif row.gol_home < row.gol_away:
            return '2'
        else:
            return 'X'
    df['actual'] = df.apply(actual_sign, axis=1)

    # predicted sign
    def predicted(row):
        t = row.BSV_type
        if t == 'SEMI_BVS' or t == 'NO_BVS':
            return None

        # Ordine delle quote bookmaker
        bk_order = order_signs(row.bk_q1, row.bk_qx, row.bk_q2)
        bk_fav = bk_order[0] if len(bk_order) > 0 else None

        # BVS_1X e BVS_X2 agganciati a pattern di "scala" sulle quote book
        # (non usiamo il picchetto qui perché è già incorporato nella logica BVS).
        try:
            q1 = float(row.bk_q1)
            qx = float(row.bk_qx)
            q2 = float(row.bk_q2)
        except Exception:
            q1 = qx = q2 = None

        # pattern scala 1X2 sui bookmaker: q1 < qx < q2
        scala_1x2 = q1 is not None and qx is not None and q2 is not None and (q1 < qx < q2)
        # pattern scala 2X1 sui bookmaker: q2 < qx < q1
        scala_2x1 = q1 is not None and qx is not None and q2 is not None and (q2 < qx < q1)

        if t == 'BVS_1X':
            # prediciamo l'1 solo se la struttura delle quote supporta davvero 1X2
            return '1' if scala_1x2 else None
        if t == 'BVS_X2':
            # prediciamo il 2 solo se la struttura delle quote supporta davvero 2X1
            return '2' if scala_2x1 else None

        # tutti gli altri BVS usano il segno favorito dal bookmaker
        return bk_fav
    df['pred'] = df.apply(predicted, axis=1)

    # accuracy per group (con numero di occorrenze)
    print("Accuracy per BSV_type (con occorrenze):")
    for t, g in df.groupby("BSV_type"):
        total_count = len(g)
        valid = g.dropna(subset=["pred"])
        valid_count = len(valid)

        if valid_count > 0:
            acc = (valid["pred"] == valid["actual"]).mean()
            print(f"  {t:<20} tot={total_count:6}  con_pred={valid_count:6}  acc={acc:0.3f}")
        else:
            print(f"  {t:<20} tot={total_count:6}  con_pred=0       acc=NA (nessuna predizione)")

    # -------------------------------------------------
    # Analisi SEMI_BVS: quanto indoviniamo il segno 1/2
    # quando la favorita comune (book + picchetto) è 1 o 2
    # -------------------------------------------------
    print("\nAnalisi SEMI_BVS (favorita 1 o 2):")
    semi = df[df["BSV_type"] == "SEMI_BVS"].copy()

    if len(semi) == 0:
        print("  Nessuna partita SEMI_BVS trovata.")
    else:
        def fav_sign(row):
            # favorita del book (e del picchetto, per definizione SEMI_BVS)
            signs = order_signs(row.bk_q1, row.bk_qx, row.bk_q2)
            return signs[0] if len(signs) > 0 else None

        semi["fav_sign"] = semi.apply(fav_sign, axis=1)

        # Consideriamo solo casi con favorita 1 o 2 (escludiamo X favorita)
        semi_valid = semi[semi["fav_sign"].isin(["1", "2"])].copy()

        total_semi_1 = len(semi_valid[semi_valid["fav_sign"] == "1"])
        total_semi_2 = len(semi_valid[semi_valid["fav_sign"] == "2"])

        hit_semi_1 = int((semi_valid[semi_valid["fav_sign"] == "1"]["actual"] == "1").sum())
        hit_semi_2 = int((semi_valid[semi_valid["fav_sign"] == "2"]["actual"] == "2").sum())

        rate_1 = hit_semi_1 / total_semi_1 if total_semi_1 > 0 else 0.0
        rate_2 = hit_semi_2 / total_semi_2 if total_semi_2 > 0 else 0.0

        print(f"  SEMI_BVS favorita=1: {hit_semi_1}/{total_semi_1} = {rate_1:.3f} ({rate_1*100:.1f}%)")
        print(f"  SEMI_BVS favorita=2: {hit_semi_2}/{total_semi_2} = {rate_2:.3f} ({rate_2*100:.1f}%)")

        total_all = total_semi_1 + total_semi_2
        hits_all = hit_semi_1 + hit_semi_2
        rate_all = hits_all / total_all if total_all > 0 else 0.0
        print(f"  Totale SEMI_BVS (1 o 2 favorita): {hits_all}/{total_all} = {rate_all:.3f} ({rate_all*100:.1f}%)")

    # -------------------------------------------------
    # Analisi "Formula 4"
    # -------------------------------------------------
    # Definizione:
    #   - se 1 e X gravitano intorno a 4 → OVER 1.5 + GOL CASA SÌ
    #   - se X e 2 gravitano intorno a 4 → OVER 1.5 + GOL OSPITE SÌ
    # Intervallo "intorno a 4": [3.75, 5.25]
    print("\nAnalisi Formula 4:")

    df["F4_home"] = df.apply(is_formula4_home, axis=1)
    df["F4_away"] = df.apply(is_formula4_away, axis=1)
    df["F4_any"] = df["F4_home"] | df["F4_away"]

    # totale partite Formula 4
    f4_total = int(df["F4_any"].sum())

    if f4_total == 0:
        print("  Nessuna partita soddisfa i criteri Formula 4 (intervallo [3.75, 5.25]).")
    else:
        # Over 1.5 (almeno 2 gol totali)
        over15_hits = df.loc[df["F4_any"], :].assign(
            total_goals=lambda x: x.gol_home + x.gol_away
        )
        over15_ok = int((over15_hits["total_goals"] >= 2).sum())
        over15_rate = over15_ok / f4_total

        print(f"  Partite Formula 4 totali: {f4_total}")
        print(f"  Over 1.5 centrati: {over15_ok}/{f4_total} = {over15_rate:.3f} ({over15_rate*100:.1f}%)")

        # Gol casa sì (solo pattern 1+X)
        f4_home_df = df[df["F4_home"]].copy()
        n_home = len(f4_home_df)
        if n_home > 0:
            home_ok = int((f4_home_df["gol_home"] >= 1).sum())
            home_rate = home_ok / n_home
            print(f"  Formula 4 (1+X ~4): {n_home} partite")
            print(f"    Gol CASA sì previsti: {home_ok}/{n_home} = {home_rate:.3f} ({home_rate*100:.1f}%)")
        else:
            print("  Formula 4 (1+X ~4): nessuna partita trovata.")

        # Gol ospite sì (solo pattern X+2)
        f4_away_df = df[df["F4_away"]].copy()
        n_away = len(f4_away_df)
        if n_away > 0:
            away_ok = int((f4_away_df["gol_away"] >= 1).sum())
            away_rate = away_ok / n_away
            print(f"  Formula 4 (X+2 ~4): {n_away} partite")
            print(f"    Gol OSPITE sì previsti: {away_ok}/{n_away} = {away_rate:.3f} ({away_rate*100:.1f}%)")
        else:
            print("  Formula 4 (X+2 ~4): nessuna partita trovata.")

    # -------------------------------------------------
    # Analisi Scala 1X2 / Scala 2X1
    # -------------------------------------------------
    print("\nAnalisi Scala 1X2 / Scala 2X1:")

    def scala_prediction(row):
        # Restrizione: quota della favorita bookmaker non deve superare 1.90
        try:
            fav = order_signs(row.bk_q1, row.bk_qx, row.bk_q2)[0]
            fav_quote = {'1': row.bk_q1, 'X': row.bk_qx, '2': row.bk_q2}[fav]
            if float(fav_quote) > 1.90:
                return None
        except Exception:
            return None

        # 1X2: qb1 < qbx < qb2 AND qt1 < qtx < qt2
        if (
            row.bk_q1 < row.bk_qx < row.bk_q2 and
            row.pt_q1 < row.pt_qx < row.pt_q2
        ):
            return "SCALA_1X2_pred1"   # predicted 1

        # 2X1: qb1 > qbx > qb2 AND qt1 > qtx > qt2
        if (
            row.bk_q1 > row.bk_qx > row.bk_q2 and
            row.pt_q1 > row.pt_qx > row.pt_q2
        ):
            return "SCALA_2X1_pred2"   # predicted 2

        return None

    df["SCALA_pred"] = df.apply(scala_prediction, axis=1)

    # Count occurrences
    total_scala_1 = (df["SCALA_pred"] == "SCALA_1X2_pred1").sum()
    total_scala_2 = (df["SCALA_pred"] == "SCALA_2X1_pred2").sum()

    print(f"  Scala 1X2 occorrenze: {total_scala_1}")
    print(f"  Scala 2X1 occorrenze: {total_scala_2}")

    # Accuracy
    if total_scala_1 > 0:
        hits1 = ((df["SCALA_pred"] == "SCALA_1X2_pred1") & (df["actual"] == "1")).sum()
        acc1 = hits1 / total_scala_1
        print(f"  Scala 1X2 accuracy: {hits1}/{total_scala_1} = {acc1:.3f} ({acc1*100:.1f}%)")
    else:
        print("  Scala 1X2: nessuna occorrenza.")

    if total_scala_2 > 0:
        hits2 = ((df["SCALA_pred"] == "SCALA_2X1_pred2") & (df["actual"] == "2")).sum()
        acc2 = hits2 / total_scala_2
        print(f"  Scala 2X1 accuracy: {hits2}/{total_scala_2} = {acc2:.3f} ({acc2*100:.1f}%)")
    else:
        print("  Scala 2X1: nessuna occorrenza.")


    # -------------------------------------------------
    # MG stats for SCALA 1X2 / 2X1
    # -------------------------------------------------
    print("\nAnalisi MG per Scala 1X2 / Scala 2X1:")

    def mg_stats_scala(sub, side_prefix):
        total = len(sub)
        if total == 0:
            print(f"  Nessuna partita per {side_prefix}")
            return
        if side_prefix == "home":
            goals = sub.gol_home
        else:
            goals = sub.gol_away
        mg13 = ((goals >= 1) & (goals <= 3)).mean()
        mg14 = ((goals >= 1) & (goals <= 4)).mean()
        mg15 = (goals >= 1).mean()
        print(f"  {side_prefix} total={total}")
        print(f"    MG 1–3: {mg13:.3f} ({mg13*100:.1f}%)")
        print(f"    MG 1–4: {mg14:.3f} ({mg14*100:.1f}%)")
        print(f"    MG 1–5: {mg15:.3f} ({mg15*100:.1f}%)")

    scala1_df = df[df["SCALA_pred"] == "SCALA_1X2_pred1"]
    scala2_df = df[df["SCALA_pred"] == "SCALA_2X1_pred2"]

    print("  Scala 1X2:")
    mg_stats_scala(scala1_df, "home")

    print("  Scala 2X1:")
    mg_stats_scala(scala2_df, "away")

    # Salvo
    df.to_parquet(OUT_FILE, index=False)

    print(f"💾 Salvato in: {OUT_FILE}")
    print("📊 Distribuzione BSV_type:")
    for t, cnt in df["BSV_type"].value_counts().items():
        print(f"  {t:<20} {cnt:6}")
    print(f"\n✅ Finiti. Output: {OUT_FILE}")

    # -------------------------------------------------
    # Analisi "Metrica Forbice"
    # -------------------------------------------------
    print("\nAnalisi Metrica Forbice:")
    df["FORBICE"] = df.apply(is_forbice, axis=1)
    f_total = int(df["FORBICE"].sum())
    if f_total == 0:
        print("  Nessuna partita soddisfa la Metrica Forbice.")
    else:
        sub = df[df["FORBICE"]].copy()
        sub["total_goals"] = sub.gol_home + sub.gol_away
        o15 = int((sub["total_goals"] >= 2).sum())
        o15_rate = o15 / f_total
        o25 = int((sub["total_goals"] >= 3).sum())
        o25_rate = o25 / f_total
        print(f"  Partite Forbice: {f_total}")
        print(f"  Over 1.5: {o15}/{f_total} = {o15_rate:.3f} ({o15_rate*100:.1f}%)")
        print(f"  Over 2.5: {o25}/{f_total} = {o25_rate:.3f} ({o25_rate*100:.1f}%)")


    # -------------------------------------------------
    # Analisi MG per BVS non lineare
    # -------------------------------------------------
    print("\nAnalisi MG per BVS non lineare:")

    def mg_stats(sub, side_prefix):
        total = len(sub)
        if total == 0:
            print(f"  Nessuna partita per {side_prefix}")
            return
        if side_prefix == "home":
            goals = sub.gol_home
        else:
            goals = sub.gol_away
        mg13 = ((goals >= 1) & (goals <= 3)).mean()
        mg14 = ((goals >= 1) & (goals <= 4)).mean()
        mg15 = (goals >= 1).mean()
        print(f"  {side_prefix} total={total}")
        print(f"    MG 1–3: {mg13:.3f} ({mg13*100:.1f}%)")
        print(f"    MG 1–4: {mg14:.3f} ({mg14*100:.1f}%)")
        print(f"    MG 1–5: {mg15:.3f} ({mg15*100:.1f}%)")

    sub_home = df[df["BSV_type"] == "BVS_non_lineare_1"]
    sub_away = df[df["BSV_type"] == "BVS_non_lineare_2"]

    mg_stats(sub_home, "home")
    mg_stats(sub_away, "away")

    # -------------------------------------------------
    # Analisi metriche sperimentali (letteratura-inspired)
    # -------------------------------------------------
    print("\nAnalisi metriche sperimentali:")

    # 1) Fibonacci Odds – double chance contro favorita
    fib_mask = df.apply(is_fibonacci_odds, axis=1)
    fib_total = int(fib_mask.sum())
    print(f"\n[Metriche] Fibonacci Odds: partite rilevate = {fib_total}")
    if fib_total > 0:
        fib_df = df[fib_mask].copy()

        def fib_dc_hit(row):
            # favorita del book
            fav = order_signs(row.bk_q1, row.bk_qx, row.bk_q2)[0]
            # double chance CONTRO favorita: vince se esito != fav
            return row.actual != fav

        hits_dc = int(fib_df.apply(fib_dc_hit, axis=1).sum())
        rate_dc = hits_dc / fib_total
        print(f"  Double chance contro favorita (X2 se fav=1, 1X se fav=2): "
              f"{hits_dc}/{fib_total} = {rate_dc:.3f} ({rate_dc*100:.1f}%)")
    else:
        print("  Nessuna partita con pattern Fibonacci Odds trovata.")

    # 2) Elasticità 20% – partite equilibrate, focus su X
    el_mask = df.apply(is_elastic_20, axis=1)
    el_total = int(el_mask.sum())
    print(f"\n[Metriche] Elasticità 20%: partite rilevate = {el_total}")
    if el_total > 0:
        el_df = df[el_mask].copy()
        draws = int((el_df["actual"] == "X").sum())
        rate_draw = draws / el_total
        print(f"  Esito X: {draws}/{el_total} = {rate_draw:.3f} ({rate_draw*100:.1f}%)")
    else:
        print("  Nessuna partita in contesto di forte equilibrio (elasticità <= 20%).")

    # 3) Disordine del mercato – X quota più alta
    dis_mask = df.apply(is_disorder_x_high, axis=1)
    dis_total = int(dis_mask.sum())
    print(f"\n[Metriche] Disordine del mercato (X quota più alta): partite rilevate = {dis_total}")
    if dis_total > 0:
        dis_df = df[dis_mask].copy()
        non_draws = int((dis_df["actual"] != "X").sum())
        rate_non_draw = non_draws / dis_total
        print(f"  Esiti SECCHI (1 o 2): {non_draws}/{dis_total} = {rate_non_draw:.3f} ({rate_non_draw*100:.1f}%)")
    else:
        print("  Nessuna partita con X quota più alta trovata.")

    # 4) Doppio Gradiente – profilo MG sulla favorita
    print("\n[Metriche] Doppio Gradiente (WEAK / STRONG):")
    df["DG_type"] = df.apply(classify_double_gradient, axis=1)

    for dg_type in ["WEAK", "STRONG"]:
        sub = df[df["DG_type"] == dg_type].copy()
        n = len(sub)
        print(f"  DG_{dg_type}: partite = {n}")
        if n == 0:
            continue

        # determina lato favorita e gol favorita
        def fav_goals(row):
            fav = order_signs(row.bk_q1, row.bk_qx, row.bk_q2)[0]
            if fav == "1":
                return row.gol_home
            if fav == "2":
                return row.gol_away
            return np.nan

        sub["fav_goals"] = sub.apply(fav_goals, axis=1)
        sub = sub.dropna(subset=["fav_goals"])
        n_fg = len(sub)
        if n_fg == 0:
            print("    Nessuna partita con favorita 1 o 2 (X favorita rara).")
            continue

        goals = sub["fav_goals"]
        mg13 = ((goals >= 1) & (goals <= 3)).mean()
        mg14 = ((goals >= 1) & (goals <= 4)).mean()
        mg15 = (goals >= 1).mean()

        print(f"    Partite con fav 1/2: {n_fg}")
        print(f"    MG Fav 1–3: {mg13:.3f} ({mg13*100:.1f}%)")
        print(f"    MG Fav 1–4: {mg14:.3f} ({mg14*100:.1f}%)")
        print(f"    MG Fav 1–5: {mg15:.3f} ({mg15*100:.1f}%)")


    # 5) Compressione dei segni – std quote bassa
    comp_mask = df.apply(is_compression_quotes, axis=1)
    comp_total = int(comp_mask.sum())
    print(f"\n[Metriche] Compressione dei segni: partite rilevate = {comp_total}")
    if comp_total > 0:
        comp_df = df[comp_mask].copy()
        comp_df["total_goals"] = comp_df["gol_home"] + comp_df["gol_away"]
        draws = int((comp_df["actual"] == "X").sum())
        rate_draw = draws / comp_total
        under25 = int((comp_df["total_goals"] <= 2).sum())
        rate_under25 = under25 / comp_total
        print(f"  Esito X: {draws}/{comp_total} = {rate_draw:.3f} ({rate_draw*100:.1f}%)")
        print(f"  Under 2.5: {under25}/{comp_total} = {rate_under25:.3f} ({rate_under25*100:.1f}%)")
    else:
        print("  Nessuna partita in forte compressione delle quote (std < 0.45).")

    # 6) Book Shape
    bs_mask = df.apply(is_book_shape, axis=1)
    bs_total = int(bs_mask.sum())
    print(f"\n[Metriche] Book Shape: partite = {bs_total}")
    if bs_total > 0:
        bs_df = df[bs_mask].copy()
        draws = (bs_df['actual'] == 'X').mean()
        print(f"  Draw rate: {draws:.3f} ({draws*100:.1f}%)")

    # 7) Draw Magnet
    dm_mask = df.apply(is_draw_magnet, axis=1)
    dm_total = int(dm_mask.sum())
    print(f"\n[Metriche] Draw Magnet: partite = {dm_total}")
    if dm_total > 0:
        dm_df = df[dm_mask].copy()
        draws = (dm_df['actual']=='X').mean()
        print(f"  Draw rate: {draws:.3f} ({draws*100:.1f}%)")

    # 8) Balanced Shape
    bal_mask = df.apply(is_balanced_shape, axis=1)
    bal_total = int(bal_mask.sum())
    print(f"\n[Metriche] Balanced Shape: partite = {bal_total}")
    if bal_total > 0:
        bal_df = df[bal_mask].copy()
        o25 = ((bal_df.gol_home + bal_df.gol_away)>=3).mean()
        print(f"  Over 2.5: {o25:.3f} ({o25*100:.1f}%)")


    # 9) Double Favorite Compression (DFC)
    dfc_mask = df.apply(is_double_favorite_compression, axis=1)
    dfc_total = int(dfc_mask.sum())
    print(f"\n[Metriche] DFC: partite = {dfc_total}")
    if dfc_total > 0:
        dfc_df = df[dfc_mask].copy()
        goals = (dfc_df.gol_home + dfc_df.gol_away)
        o25 = (goals>=3).mean()
        print(f"  Over 2.5: {o25:.3f} ({o25*100:.1f}%)")


    # -------------------------------------------------
    # New Experimental Metrics: Volatility & Inverse Favorite Bias
    # -------------------------------------------------
    print("\n[Metriche] Volatilità Quote:")
    vol_mask = df.apply(is_volatility_pattern, axis=1)
    vol_total = int(vol_mask.sum())
    print(f"  Partite = {vol_total}")
    if vol_total > 0:
        vol_df = df[vol_mask].copy()
        over25 = ((vol_df.gol_home + vol_df.gol_away) >= 3).mean()
        print(f"  Over 2.5 = {over25:.3f} ({over25*100:.1f}%)")


    print("\n[Metriche] Inverse Favorite Bias:")
    ifb_mask = df.apply(is_inverse_favorite_bias, axis=1)
    ifb_total = int(ifb_mask.sum())
    print(f"  Partite = {ifb_total}")
    if ifb_total > 0:
        ifb_df = df[ifb_mask].copy()
        upsets = (ifb_df.actual != ifb_df.apply(lambda r: order_signs(r.bk_q1,r.bk_qx,r.bk_q2)[0], axis=1)).mean()
        print(f"  Upset rate (vittorie non favorita): {upsets:.3f} ({upsets*100:.1f}%)")


if __name__ == "__main__":
    main()
