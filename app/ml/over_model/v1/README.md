# OverModelV1 (Baseline)

**Input:** solo O/U 2.5 (Avg>2.5, Avg<2.5) + `league_code` opzionale  
**Output:** p(>0.5), p(>1.5), p(>2.5) calibrate + EV(O2.5)

## Pipeline
1) Rimozione vig → `p_over25_fair`
2) Inversione Poisson → `λ` con P(TotalGoals>2.5)=`p_over25_fair`
3) Poisson → `p05,p15,p25` grezze
4) Calibrazione isotonic:
   - per lega se disponibili i `.pkl`
   - fallback global `.pkl`

## Uso
```python
from app.ml.over_model.v1.model import OverModelV1

model = OverModelV1(debug=False)
row = {"Avg>2.5": 1.91, "Avg<2.5": 1.81, "league_code":"I1"}
pred = model.predict(row)
print(pred["p_over_2_5"], pred["ev_over_2_5"])
