"""Prediction + explanation (no SHAP needed)."""
import joblib
import pandas as pd

_b = joblib.load("risk_model.joblib")
MODEL, FEATURES, BASE = _b["model"], _b["features"], _b["baseline"]
CLASSES = list(MODEL.classes_)

def _index(probs):
    """0..100 risk index: Medium counts half, High counts full."""
    d = dict(zip(CLASSES, probs))
    return 100 * (0.5 * d.get("Medium", 0) + d.get("High", 0))

def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES]
    P = MODEL.predict_proba(X)
    out = df.copy()
    raw_preds = MODEL.predict(X)
    indices = [round(_index(p), 1) for p in P]
    
    # Assign consistent risk_level aligned with continuous risk index
    levels = []
    for idx, raw_p in zip(indices, raw_preds):
        if idx >= 65.0:
            levels.append("High")
        elif idx >= 30.0:
            levels.append("Medium")
        else:
            levels.append("Low")
            
    out["risk_level"] = levels
    out["risk_index"] = indices
    return out

def explain(row: pd.Series) -> pd.Series:
    """Drop each feature to the 'typical' value and see how much the risk index falls."""
    x = pd.DataFrame([row[FEATURES]])
    base_idx = _index(MODEL.predict_proba(x)[0])
    contrib = {}
    for f in FEATURES:
        x2 = x.copy()
        x2[f] = BASE[f]
        contrib[f] = base_idx - _index(MODEL.predict_proba(x2)[0])
    return pd.Series(contrib).sort_values(ascending=False)
