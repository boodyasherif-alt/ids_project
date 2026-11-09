import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IDS Alert Dashboard", layout="wide")
st.title("🛡️ ML-based Intrusion Detection System")
st.caption("Model: XGBoost trained on CICIDS2017 (Wednesday + Friday DDoS). "
           "Demo replays recorded flows, not live traffic.")

@st.cache_resource
def load_model():
    return (joblib.load("xgb_model.joblib"),
            joblib.load("label_encoder.joblib"),
            joblib.load("feature_cols.joblib"))

@st.cache_data
def load_flows():
    return pd.read_csv("demo_flows.csv")

model, le, feature_cols = load_model()
flows = load_flows()

# Classify all flows
X = flows[feature_cols].astype("float32")
proba = model.predict_proba(X)
flows["Predicted"] = le.inverse_transform(proba.argmax(axis=1))
flows["Confidence"] = proba.max(axis=1).round(3)

alerts = flows[flows["Predicted"] != "BENIGN"].copy()

# Summary numbers
c1, c2, c3 = st.columns(3)
c1.metric("Flows analyzed", f"{len(flows):,}")
c2.metric("Alerts raised", f"{len(alerts):,}")
c3.metric("Unique attacker IPs", alerts["Source IP"].nunique())

# Charts
left, right = st.columns(2)
left.subheader("Top attacker IPs")
left.bar_chart(alerts["Source IP"].value_counts().head(5))
right.subheader("Attack types")
right.bar_chart(alerts["Predicted"].value_counts())

# Filter and table
st.subheader("Alerts")
types = st.multiselect("Filter by attack type",
                       sorted(alerts["Predicted"].unique()),
                       default=sorted(alerts["Predicted"].unique()))
shown = alerts[alerts["Predicted"].isin(types)]
st.dataframe(
    shown[["Timestamp", "Source IP", "Destination IP", "Predicted", "Confidence", "Label"]]
    .rename(columns={"Label": "True label"}),
    use_container_width=True, height=400)


# ---------------- Replay + Discord alerts ----------------
import os, time, requests

WEBHOOK = os.environ.get("DISCORD_WEBHOOK")

def send_alert(text):
    if not WEBHOOK:
        return False
    try:
        r = requests.post(WEBHOOK, json={"content": text}, timeout=10)
        return r.ok
    except Exception:
        return False

st.subheader("▶ Live replay (simulated)")
st.caption("Replays recorded flows one by one. One Discord notification is sent per new "
           "(attacker IP, attack type) pair. Source IPs come from the recorded dataset.")

n = st.slider("Number of alerts to replay", 5, 100, 20)
delay = st.slider("Delay between alerts (seconds)", 0.1, 2.0, 0.5)
use_notify = st.checkbox("Send Discord notifications", value=bool(WEBHOOK))
if use_notify and not WEBHOOK:
    st.warning("DISCORD_WEBHOOK not set in this terminal. Notifications will be skipped.")

if st.button("Start replay"):
    seen = set()
    box = st.empty()
    log = []
    for _, row in alerts.sort_values("Timestamp").head(n).iterrows():
        key = (row["Source IP"], row["Predicted"])
        log.append({
            "Time": row["Timestamp"], "Attacker": row["Source IP"],
            "Target": row["Destination IP"], "Attack": row["Predicted"],
            "Confidence": row["Confidence"],
        })
        box.dataframe(pd.DataFrame(log).iloc[::-1], use_container_width=True)
        if use_notify and key not in seen:
            seen.add(key)
            send_alert(
                f"🚨 ALERT: {row['Predicted']}\n"
                f"Attacker: {row['Source IP']} -> {row['Destination IP']}\n"
                f"Confidence: {row['Confidence']}\n"
                f"Time: {row['Timestamp']}\n"
                f"(replayed from recorded dataset)")
        time.sleep(delay)
    st.success(f"Replay finished. {len(seen)} notification(s) sent.")