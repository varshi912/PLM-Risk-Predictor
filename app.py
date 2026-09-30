"""Step 4 - Streamlit dashboard. Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st
import risk_engine as eng
import windchill_client as wc

st.set_page_config(page_title="ECR Risk Predictor", layout="wide", page_icon="⚙️")
st.title("⚙️ AI-Powered Engineering Change Risk Predictor")
st.caption("Reads change requests from Windchill PLM (read-only) and evaluates engineering risk.")

src = st.sidebar.radio("Data source", ["Windchill REST", "Upload CSV", "Demo data"])
if "raw" not in st.session_state:
    st.session_state.raw = None

if src == "Windchill REST":
    url = st.sidebar.text_input("Windchill URL", "http://plm.cit.com")
    user = st.sidebar.text_input("Username")
    pwd = st.sidebar.text_input("Password", type="password")
    entity = st.sidebar.selectbox("Object type", ["ChangeRequests", "ChangeNotices"])
    verify = st.sidebar.checkbox("Verify SSL certificate", value=False)
    if st.sidebar.button("Fetch from Windchill", use_container_width=True):
        try:
            st.session_state.raw = wc.to_features(wc.fetch_changes(url, user, pwd, entity, verify=verify))
            st.success(f"Successfully loaded {len(st.session_state.raw)} records from Windchill")
        except Exception as e:
            st.error(f"Could not read Windchill: {e}")
elif src == "Upload CSV":
    up = st.sidebar.file_uploader("CSV with the model columns", type="csv")
    if up:
        df = pd.read_csv(up)
        for c, v in wc.DEFAULTS.items():
            if c not in df:
                df[c] = v
        if "number" not in df:
            df.insert(0, "number", range(1, len(df) + 1))
        if "name" not in df:
            df.insert(1, "name", "")
        st.session_state.raw = df
else:
    d = pd.read_csv("ecr_history.csv").sample(15, random_state=1).reset_index(drop=True)
    d.insert(0, "number", [f"DEMO-{i+1:03d}" for i in range(len(d))])
    d.insert(1, "name", "")
    st.session_state.raw = d.drop(columns=["risk_level"])

if st.session_state.raw is None:
    st.info("👈 Choose a data source in the sidebar to begin.")
    st.stop()

st.subheader("1. Change Requests (Interactive Table)")
st.caption("💡 Edit any parameter below (e.g. priority, cost, supplier impact) to simulate risk changes.")
edited = st.data_editor(
    st.session_state.raw,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "change_type": st.column_config.SelectboxColumn(
            "Change Type",
            options=["Design", "Material", "Supplier", "Process", "Software", "Documentation"],
            required=True,
        ),
        "priority": st.column_config.SelectboxColumn(
            "Priority",
            options=["Low", "Medium", "High", "Critical"],
            required=True,
        ),
        "supplier_impact": st.column_config.SelectboxColumn(
            "Supplier Impact",
            options=[0, 1],
            help="0 = No, 1 = Yes"
        ),
        "tooling_change": st.column_config.SelectboxColumn(
            "Tooling Change",
            options=[0, 1],
            help="0 = No, 1 = Yes"
        ),
        "regulatory_impact": st.column_config.SelectboxColumn(
            "Regulatory Impact",
            options=[0, 1],
            help="0 = No, 1 = Yes"
        )
    }
)

scored = eng.score_frame(edited)

def style_risk(val):
    if val == "Low":
        return "background-color: rgba(34, 197, 94, 0.25); color: #4ade80; font-weight: 700; border-radius: 4px; padding: 3px 8px;"
    elif val == "Medium":
        return "background-color: rgba(234, 179, 8, 0.25); color: #fde047; font-weight: 700; border-radius: 4px; padding: 3px 8px;"
    elif val == "High":
        return "background-color: rgba(239, 68, 68, 0.25); color: #f87171; font-weight: 700; border-radius: 4px; padding: 3px 8px;"
    return ""

st.subheader("2. AI Risk Predictions")
c1, c2, c3 = st.columns(3)
c1.metric("🟢 Low Risk", int((scored.risk_level == "Low").sum()))
c2.metric("🟡 Medium Risk", int((scored.risk_level == "Medium").sum()))
c3.metric("🔴 High Risk", int((scored.risk_level == "High").sum()))

show = scored[["number", "name", "change_type", "priority", "risk_level", "risk_index"]].copy()
show["risk_index"] = show["risk_index"].round(1)

st.dataframe(
    show.style.map(style_risk, subset=["risk_level"]),
    column_config={
        "risk_index": st.column_config.NumberColumn(
            "Risk Index (0-100)",
            help="Continuous risk score calculated by AI model",
            format="%.1f",
        ),
        "risk_level": st.column_config.TextColumn(
            "Risk Level",
            help="Categorical risk rating (Low, Medium, High)"
        )
    },
    use_container_width=True
)

st.subheader("3. Why this score? (Explainable AI)")
pick = st.selectbox(
    "Select a change request to inspect:",
    scored.index,
    format_func=lambda i: f"{scored.loc[i, 'number']} - {scored.loc[i, 'name']} ({scored.loc[i, 'risk_level']} Risk)"
)

row = scored.loc[pick]
st.write(f"### Assessment: **{row.risk_level} Risk** (Risk Index: `{row.risk_index} / 100`)")
ex = eng.explain(row)
st.bar_chart(ex.head(6))
st.caption("📊 Bars show how many risk-index points each factor adds compared to a standard baseline change.")
