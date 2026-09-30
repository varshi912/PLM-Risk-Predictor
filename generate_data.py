"""Step 1 - create a synthetic ECR/ECO history (replace with real data if you get it)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000

CHANGE_TYPES = ["Design", "Material", "Supplier", "Process", "Software", "Documentation"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]

df = pd.DataFrame({
    "change_type": rng.choice(CHANGE_TYPES, N, p=[.30, .15, .12, .15, .10, .18]),
    "priority": rng.choice(PRIORITIES, N, p=[.25, .40, .25, .10]),
    "affected_parts": rng.integers(1, 60, N),
    "affected_assemblies": rng.integers(0, 15, N),
    "estimated_cost": rng.integers(500, 150000, N),
    "lead_time_days": rng.integers(2, 120, N),
    "supplier_impact": rng.integers(0, 2, N),
    "tooling_change": rng.integers(0, 2, N),
    "regulatory_impact": rng.choice([0, 1], N, p=[.85, .15]),
    "requester_experience_yrs": rng.integers(0, 20, N),
    "prior_similar_changes": rng.integers(0, 10, N),
})

type_w = {"Design": 1.0, "Material": 1.5, "Supplier": 1.8, "Process": 1.4, "Software": 1.2, "Documentation": 0.2}
prio_w = {"Low": 0.0, "Medium": 0.5, "High": 1.0, "Critical": 1.6}

score = (
    df.change_type.map(type_w) + df.priority.map(prio_w)
    + df.affected_parts / 15 + df.affected_assemblies / 5
    + df.estimated_cost / 50000
    + (30 - df.lead_time_days.clip(upper=30)) / 15      # short lead time = more risk
    + 1.2 * df.supplier_impact + 1.0 * df.tooling_change + 1.5 * df.regulatory_impact
    - df.requester_experience_yrs / 15 - 0.15 * df.prior_similar_changes
    + rng.normal(0, 0.6, N)                              # noise so the model is not trivial
)
q1, q2 = score.quantile([.45, .80])
df["risk_level"] = np.where(score < q1, "Low", np.where(score < q2, "Medium", "High"))

df.to_csv("ecr_history.csv", index=False)
print("Synthetic data generated successfully:")
print(df.risk_level.value_counts())
