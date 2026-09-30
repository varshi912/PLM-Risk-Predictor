"""Step 2 - train the ensemble model and save it."""
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CAT = ["change_type", "priority"]
NUM = [
    "affected_parts", "affected_assemblies", "estimated_cost", "lead_time_days",
    "supplier_impact", "tooling_change", "regulatory_impact",
    "requester_experience_yrs", "prior_similar_changes"
]

df = pd.read_csv("ecr_history.csv")
X, y = df[CAT + NUM], df["risk_level"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough")
model = Pipeline([
    ("pre", pre),
    ("rf", RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42))
])
model.fit(Xtr, ytr)

pred = model.predict(Xte)
print("Classification Report:")
print(classification_report(yte, pred))
print("Confusion Matrix (Low, Medium, High):")
print(confusion_matrix(yte, pred, labels=["Low", "Medium", "High"]))

# baseline (typical change) used by the explanation function
baseline = {c: df[c].mode()[0] for c in CAT}
baseline.update({c: float(df[c].median()) for c in NUM})
joblib.dump({"model": model, "features": CAT + NUM, "cat": CAT, "num": NUM, "baseline": baseline},
            "risk_model.joblib")
print("saved risk_model.joblib")
