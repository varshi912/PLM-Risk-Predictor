"""Step 3 - read-only connector to Windchill REST Services (WRS)."""
import requests
import pandas as pd
from datetime import datetime, timezone
from urllib.parse import urlparse

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
    p = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    root = f"{p.scheme}://{p.netloc}" if p.netloc else base_url.rstrip('/')
    url = f"{root}/Windchill/servlet/odata/ChangeMgmt/{entity}"
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


def _infer_from_text(name: str, desc: str = "") -> dict:
    """Intelligently infer engineering attributes from change title/description when raw PLM fields are unassigned."""
    txt = f"{name} {desc}".lower()
    inferred = {}
    
    # Priority heuristics
    if any(k in txt for k in ["failuer", "failure", "fatigue", "crack", "blow hole", "safety", "urgent", "critical", "hazard"]):
        inferred["priority"] = "Critical" if any(k in txt for k in ["failuer", "failure", "crack", "safety", "urgent"]) else "High"
    elif any(k in txt for k in ["high", "defect", "flaw", "strength", "overheat", "overload"]):
        inferred["priority"] = "High"
    elif any(k in txt for k in ["minor", "draft", "note", "cosmetic", "drawing"]):
        inferred["priority"] = "Low"
        
    # Change Type heuristics
    if any(k in txt for k in ["mould", "mold", "tool", "die", "fixture", "weld", "melting", "temp", "cast", "machining", "blow hole"]):
        inferred["change_type"] = "Process"
    elif any(k in txt for k in ["alloy", "steel", "material", "crack", "fatigue", "polymer", "composite"]):
        inferred["change_type"] = "Material"
    elif any(k in txt for k in ["vendor", "supplier", "procure", "outsourc"]):
        inferred["change_type"] = "Supplier"
    elif any(k in txt for k in ["doc", "drawing", "note", "spec", "standard"]):
        inferred["change_type"] = "Documentation"
    elif any(k in txt for k in ["software", "firmware", "code", "logic", "algorithm"]):
        inferred["change_type"] = "Software"
        
    # Cost, tooling, supplier & regulatory impacts
    if any(k in txt for k in ["mould", "mold", "tool", "die", "fixture", "pattern"]):
        inferred["tooling_change"] = 1
        inferred["estimated_cost"] = 55000
    if any(k in txt for k in ["supplier", "vendor", "partner", "procure"]):
        inferred["supplier_impact"] = 1
        inferred["estimated_cost"] = 40000
    if any(k in txt for k in ["crack", "fatigue", "failuer", "failure", "safety", "hazard"]):
        inferred["regulatory_impact"] = 1
        inferred["estimated_cost"] = 75000
        inferred["lead_time_days"] = 7
        inferred["affected_parts"] = 18
        inferred["affected_assemblies"] = 4
    elif any(k in txt for k in ["blow hole", "defect", "strength"]):
        inferred["estimated_cost"] = 38000
        inferred["lead_time_days"] = 14
        inferred["affected_parts"] = 12
    elif any(k in txt for k in ["draft", "angle", "note", "cosmetic"]):
        inferred["estimated_cost"] = 3000
        inferred["lead_time_days"] = 45
        inferred["affected_parts"] = 2

    return inferred


def to_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Map Windchill columns to model inputs with smart defaults and feature extraction."""
    rows = []
    for _, r in raw.iterrows():
        f = dict(DEFAULTS)
        name_str = str(_pick(r, "Name", "Title", "ChangeName") or "")
        desc_str = str(_pick(r, "Description", "ChangeDescription") or "")
        
        # 1. Apply smart keyword heuristics from change title/description
        inferred = _infer_from_text(name_str, desc_str)
        f.update(inferred)
        
        # 2. If explicit Windchill properties are provided, they take highest priority
        cat = str(_pick(r, "Category", "ChangeCategory", "Type") or "").lower()
        if cat:
            f["change_type"] = next((v for k, v in TYPE_MAP.items() if k in cat), f["change_type"])
            
        prio = str(_pick(r, "Priority", "ChangePriority", "Severity") or "").lower()
        if prio and prio in PRIO_MAP:
            f["priority"] = PRIO_MAP[prio]
            
        need = _pick(r, "NeedDate", "NeedByDate", "TargetDate", "DueDate")
        if need:
            try:
                d = pd.to_datetime(need, utc=True)
                f["lead_time_days"] = max(1, (d - datetime.now(timezone.utc)).days)
            except Exception:
                pass
                
        n = _pick(r, "AffectedObjectCount", "AffectedCount", "AffectedItemsCount")
        if n is not None:
            try:
                f["affected_parts"] = int(n)
            except Exception:
                pass
                
        cost = _pick(r, "EstimatedCost", "Cost", "Budget")
        if cost is not None:
            try:
                f["estimated_cost"] = float(cost)
            except Exception:
                pass

        rows.append({"number": _pick(r, "Number", "ID", "ChangeNumber") or "", "name": name_str, **f})
    return pd.DataFrame(rows)
