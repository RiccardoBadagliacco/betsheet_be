# ==============================================
# app/ml/stats_gol/model.py
# ==============================================

import math
import numpy as np
from sqlalchemy import or_
from app.db.models import Match


FAV_HOME_THR = 1.71
FAV_AWAY_THR = 1.81


class MgFavModelV1:
    """Modello Multigol (Favorite-based)"""

    def __init__(self, db, window=20, debug=False):
        self.db = db
        self.window = window
        self.debug = debug


    def _team_stats(self, team_id, season_id, as_home, up_to_date, is_fixture=False):
        """
        Calcola le medie gol fatti/subiti della squadra.
        Se is_fixture=True → ignora il vincolo di stagione e considera le ultime 30 partite totali.
        Applica un peso decrescente: ultime 10 partite = 1.0, 10–20 = 0.7, 20–30 = 0.5
        """
        from sqlalchemy import or_
        from app.db.models import Match
        import numpy as np

        if is_fixture:
            q = (
                self.db.query(Match)
                .filter(or_(Match.home_team_id == team_id, Match.away_team_id == team_id))
                .filter(Match.home_goals_ft != None, Match.away_goals_ft != None)
                .filter(Match.match_date < up_to_date)
                .order_by(Match.match_date.desc())
                .limit(30)
            )
        else:
            q = (
                self.db.query(Match)
                .filter(Match.season_id == season_id)
                .filter(or_(Match.home_team_id == team_id, Match.away_team_id == team_id))
                .filter(Match.home_goals_ft != None, Match.away_goals_ft != None)
                .filter(Match.match_date < up_to_date)
                .order_by(Match.match_date.desc())
                .limit(self.window)
            )

        rows = q.all()
        if not rows:
            return {"gf": 1.2, "ga": 1.2}  # fallback neutro

        gf, ga, weights = [], [], []

        for idx, r in enumerate(rows):
            if r.home_team_id == team_id:
                gf.append(r.home_goals_ft)
                ga.append(r.away_goals_ft)
            else:
                gf.append(r.away_goals_ft)
                ga.append(r.home_goals_ft)

            # assegna peso (solo in fixture mode)
  
            if idx < 10:
                weights.append(1.0)
            elif idx < 20:
                weights.append(0.7)
            else:
                weights.append(0.5)
            

        # calcolo medie pesate
        weights = np.array(weights)
        gf_mean = np.average(gf, weights=weights)
        ga_mean = np.average(ga, weights=weights)

        if self.debug:
            mode = "FIXTURE" if is_fixture else "HIST"
            print(f"[DEBUG] {mode} | team={team_id} | n={len(rows)} | gf={gf_mean:.2f}, ga={ga_mean:.2f}")

        return {"gf": gf_mean, "ga": ga_mean}
    # ======================================================
    # Predizione multigol per un match o fixture
    # ======================================================
    def predict_for_match(self, m):
        """
        Calcola le probabilità MG 1-3, 1-4, 1-5 per il favorito.
        """
        from app.db.models import Season

        is_fixture = hasattr(m, "home_goals_ft") and m.home_goals_ft is None
        season_id = getattr(m, "season_id", None)
        up_to_date = getattr(m, "match_date")

        ht_id = getattr(m, "home_team_id", None)
        at_id = getattr(m, "away_team_id", None)

        if not ht_id or not at_id:
            return None

        # quote favorite
        fav = None
        if m.avg_home_odds and m.avg_away_odds:
            if m.avg_home_odds < FAV_HOME_THR and (m.avg_home_odds < m.avg_away_odds):
                fav = "home"
            elif m.avg_away_odds < FAV_AWAY_THR and (m.avg_away_odds < m.avg_home_odds):
                fav = "away"

        if not fav:
            return None

        # statistiche team
        s_home = self._team_stats(ht_id, season_id, as_home=True, up_to_date=up_to_date, is_fixture=is_fixture)
        s_away = self._team_stats(at_id, season_id, as_home=False, up_to_date=up_to_date, is_fixture=is_fixture)

        if fav == "home":
            λ = (s_home["gf"] + s_away["ga"]) / 2
        else:
            λ = (s_away["gf"] + s_home["ga"]) / 2

        # prob Poisson multigol cumulative
        # --- calcolo probabilità multigol (Poisson) ---
        def p_range(lam, a, b):
            import math
            p = 0.0
            for k in range(a, b + 1):
                p += math.exp(-lam) * (lam ** k) / math.factorial(k)
            return p

        # nomi senza il prefisso “MG_” per evitare doppioni nel return
        probs_named = {
            "1_3": p_range(λ, 1, 3),
            "1_4": p_range(λ, 1, 4),
            "1_5": p_range(λ, 1, 5),
        }

        if fav == "home":
            return {
                "favorite_side": "home",
                "lambda_home": λ,
                "MG_HOME_1_3": probs_named["1_3"],
                "MG_HOME_1_4": probs_named["1_4"],
                "MG_HOME_1_5": probs_named["1_5"],
            }
        else:
            return {
                "favorite_side": "away",
                "lambda_away": λ,
                "MG_AWAY_1_3": probs_named["1_3"],
                "MG_AWAY_1_4": probs_named["1_4"],
                "MG_AWAY_1_5": probs_named["1_5"],
            }
