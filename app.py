"""Step 4 - Streamlit dashboard. Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st
import risk_engine as eng
import windchill_client as wc

st.set_page_config(page_title="ECR Risk Predictor", layout="wide")
st.title("AI-Powered Engineering Change Risk Predictor")
st.caption("Reads change requests from Windchill (read-only) and scores their risk.")

src = st.sidebar.radio("Data source", ["Windchill REST", "Upload CSV", "Demo data"])
if "raw" not in st.session_state:
    st.session_state.raw = None

if src == "Windchill REST":
    url = st.sidebar.text_input("Windchill URL", "http://plm.cit.com")
    user = st.sidebar.text_input("Username")
    pwd = st.sidebar.text_input("Password", type="password")
    entity = st.sidebar.selectbox("Object type", ["ChangeRequests", "ChangeNotices"])
    verify = st.sidebar.checkbox("Verify SSL certificate", value=False)
    if st.sidebar.button("Fetch from Windchill"):
        try:
            st.session_state.raw = wc.to_features(wc.fetch_changes(url, user, pwd, entity, verify=verify))
            st.success(f"Loaded {len(st.session_state.raw)} records")
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
    st.info("Choose a data source in the sidebar.")
    st.stop()

st.subheader("1. Change requests (edit any value Windchill did not provide)")
edited = st.data_editor(st.session_state.raw, num_rows="fixed", use_container_width=True)
scored = eng.score_frame(edited)

COL = {"Low": "#c6efce", "Medium": "#ffeb9c", "High": "#ffc7ce"}
st.subheader("2. Risk scores")
c1, c2, c3 = st.columns(3)
for col, lvl in zip((c1, c2, c3), ("Low", "Medium", "High")):
    col.metric(f"{lvl} risk", int((scored.risk_level == lvl).sum()))
show = scored[["number", "name", "change_type", "priority", "risk_level", "risk_index"]]
st.dataframe(show.style.map(lambda v: f"background-color:{COL.get(v,'')}", subset=["risk_level"]),
             use_container_width=True)

st.subheader("3. Why this score?")
pick = st.selectbox("Select a change", scored.index,
                    format_func=lambda i: f"{scored.loc[i,'number']} - {scored.loc[i,'risk_level']}")
row = scored.loc[pick]
st.write(f"**{row.risk_level} risk** (index {row.risk_index}/100)")
ex = eng.explain(row)
st.bar_chart(ex.head(6))
st.caption("Bars show how many risk-index points each factor adds compared with a typical change.")
