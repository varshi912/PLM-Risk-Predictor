"""Step 3 - read-only connector to Windchill REST Services (WRS)."""
import requests
import pandas as pd
from datetime import datetime, timezone

DEFAULTS = {
    "change_type": "Design",
    "priority": "Medium",
    "affected_parts": 5,
    "affected_assemblies": 1,
    "estimated_cost": 5000,
    "lead_time_days": 30,
    "supplier_impact": 0,
    "tooling_change": 0,
    "regulatory_impact": 0,
    "requester_experience_yrs": 3,
    "prior_similar_changes": 2
}

# Windchill category text -> model change_type (edit to match your server's values)
TYPE_MAP = {
    "design": "Design",
    "material": "Material",
    "supplier": "Supplier",
    "process": "Process",
    "software": "Software",
    "documentation": "Documentation"
}
PRIO_MAP = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "critical": "Critical",
    "urgent": "Critical",
    "emergency": "Critical",
    "normal": "Medium"
}


def fetch_changes(base_url, user, password, entity="ChangeRequests", top=50, verify=True):
    """GET .../odata/ChangeMgmt/<entity>. Read-only: nothing is written to Windchill."""
    url = f"{base_url.rstrip('/')}/Windchill/servlet/odata/ChangeMgmt/{entity}"
    r = requests.get(
        url,
        params={"$top": top},
        auth=(user, password),
        headers={"Accept": "application/json"},
        verify=verify,
        timeout=30
    )
    r.raise_for_status()
    return pd.DataFrame(r.json().get("value", []))


def _pick(row, *keys):
    for k in keys:
        if k in row and pd.notna(row[k]) and row[k] != "":
            v = row[k]
            return v.get("Display", v.get("Value")) if isinstance(v, dict) else v
    return None


def to_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Map Windchill columns to model inputs. Anything missing gets a default the user can edit."""
    rows = []
    for _, r in raw.iterrows():
        f = dict(DEFAULTS)
        cat = str(_pick(r, "Category", "ChangeCategory") or "").lower()
        f["change_type"] = next((v for k, v in TYPE_MAP.items() if k in cat), f["change_type"])
        f["priority"] = PRIO_MAP.get(str(_pick(r, "Priority", "ChangePriority") or "").lower(), f["priority"])
        need = _pick(r, "NeedDate", "NeedByDate")
        if need:
            try:
                d = pd.to_datetime(need, utc=True)
                f["lead_time_days"] = max(1, (d - datetime.now(timezone.utc)).days)
            except Exception:
                pass
        n = _pick(r, "AffectedObjectCount")
        if n is not None:
            f["affected_parts"] = int(n)
        rows.append({"number": _pick(r, "Number", "ID") or "", "name": _pick(r, "Name") or "", **f})
    return pd.DataFrame(rows)
