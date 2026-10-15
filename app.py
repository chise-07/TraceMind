
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

# -----------------------------
# 1. Page settings
# -----------------------------
st.set_page_config(
    page_title="TraceMind",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ TraceMind")
st.subheader("Explainable Cyber Anomaly Detector")

st.write(
    "TraceMind uses machine learning to identify unusual login "
    "activity and explains why an event may be suspicious."
)

# -----------------------------
# 2. Generate example login data
# -----------------------------
@st.cache_data
def generate_login_data():
    random.seed(42)
    records = []
    start_time = datetime(2026, 10, 1, 8, 0)

    usernames = ["alice", "bob", "charlie", "david", "emma"]

    # Generate ordinary login activity
    for i in range(200):
        hour = random.randint(7, 18)
        minute = random.randint(0, 59)

        records.append({
            "username": random.choice(usernames),
            "login_time": (start_time + timedelta(days=i // 20)).replace(
                hour=hour, minute=minute
            ).strftime("%Y-%m-%d %H:%M"),
            "login_hour": hour,
            "failed_attempts": random.choices(
                [0, 1, 2], weights=[75, 20, 5]
            )[0],
            "login_status": random.choices(
                ["Success", "Failed"], weights=[90, 10]
            )[0]
        })

    # Add unusual events for testing
    unusual_events = [
        ("alice", 2, 15),
        ("bob", 3, 12),
        ("charlie", 1, 18),
        ("david", 4, 14),
        ("emma", 0, 16),
        ("alice", 3, 10)
    ]

    for username, hour, failures in unusual_events:
        records.append({
            "username": username,
            "login_time": (
                start_time + timedelta(days=12)
            ).replace(hour=hour, minute=15).strftime("%Y-%m-%d %H:%M"),
            "login_hour": hour,
            "failed_attempts": failures,
            "login_status": "Failed" if failures >= 3 else "Success"
        })

    return pd.DataFrame(records)


df = generate_login_data()

# -----------------------------
# 3. Machine learning detection
# -----------------------------
features = df[["login_hour", "failed_attempts"]]

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

df["prediction"] = model.fit_predict(features)

# Isolation Forest returns -1 for an anomaly and 1 for a normal event
df["AI Anomaly"] = df["prediction"].apply(
    lambda value: "Suspicious" if value == -1 else "Normal"
)

# -----------------------------
# 4. Explain the alerts
# -----------------------------
def explain_event(row):
    reasons = []

    if row["login_hour"] < 6 or row["login_hour"] > 22:
        reasons.append("Login occurred at an unusual hour")

    if row["failed_attempts"] >= 3:
        reasons.append("Several failed login attempts")

    if row["login_hour"] < 6 and row["failed_attempts"] >= 2:
        reasons.append("Combination of late login and failed attempts")

    if not reasons:
        reasons.append(
            "The AI model found this login pattern unusual"
        )

    return "; ".join(reasons)


df["Explanation"] = df.apply(explain_event, axis=1)

# -----------------------------
# 5. Dashboard summary
# -----------------------------
st.subheader("Login Activity Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Login Events", len(df))

with col2:
    st.metric("Unique Users", df["username"].nunique())

with col3:
    st.metric(
        "AI-Flagged Events",
        int((df["AI Anomaly"] == "Suspicious").sum())
    )

# -----------------------------
# 6. Filters
# -----------------------------
st.subheader("Explore Login Records")

selected_user = st.selectbox(
    "Filter by username",
    ["All users"] + sorted(df["username"].unique().tolist())
)

selected_status = st.selectbox(
    "Filter by AI result",
    ["All results", "Suspicious", "Normal"]
)

filtered_df = df.copy()

if selected_user != "All users":
    filtered_df = filtered_df[
        filtered_df["username"] == selected_user
    ]

if selected_status != "All results":
    filtered_df = filtered_df[
        filtered_df["AI Anomaly"] == selected_status
    ]

st.dataframe(
    filtered_df[
        [
            "username",
            "login_time",
            "failed_attempts",
            "login_status",
            "AI Anomaly",
            "Explanation"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# 7. Suspicious activity details
# -----------------------------
st.subheader("Suspicious Activity Details")

suspicious = filtered_df[
    filtered_df["AI Anomaly"] == "Suspicious"
]

if not suspicious.empty:
    st.warning(
        "The AI model identified login events that differ "
        "from the patterns it learned."
    )

    for _, row in suspicious.head(10).iterrows():
        st.write(
            f"**User:** {row['username']} | "
            f"**Time:** {row['login_time']} | "
            f"**Reason:** {row['Explanation']}"
        )
else:
    st.info("No AI-flagged events match the current filters.")

# -----------------------------
# 8. Chart
# -----------------------------
st.subheader("Login Activity by Hour")

hour_counts = df.groupby("login_hour").size().reset_index(
    name="Number of logins"
)

st.bar_chart(
    hour_counts.set_index("login_hour")
)

# -----------------------------
# 9. Download results
# -----------------------------
csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download analysis as CSV",
    data=csv_data,
    file_name="tracemind_analysis.csv",
    mime="text/csv"
)

st.caption(
    "Educational prototype using synthetic data. "
    "An AI alert indicates unusual behaviour, not confirmed malicious activity."
)


# -----------------------------
# 10. Model Performance Evaluation
# -----------------------------
st.subheader("AI Model Performance")

st.write(
    "These are demonstration metrics using synthetic test labels. "
    "They are not proof of real-world detection accuracy."
)

# Create example labels independently of the AI predictions.
# These simple rules are only approximate labels for this demo.
df["Known Unusual"] = (
    (df["failed_attempts"] >= 3)
    | (df["login_hour"] < 5)
).astype(int)

df["AI Prediction"] = (
    df["AI Anomaly"] == "Suspicious"
).astype(int)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

precision = precision_score(
    df["Known Unusual"],
    df["AI Prediction"],
    zero_division=0
)

recall = recall_score(
    df["Known Unusual"],
    df["AI Prediction"],
    zero_division=0
)

f1 = f1_score(
    df["Known Unusual"],
    df["AI Prediction"],
    zero_division=0
)

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Precision", f"{precision:.2%}")
metric2.metric("Recall", f"{recall:.2%}")
metric3.metric("F1-Score", f"{f1:.2%}")

st.caption(
    "The labels use simple rules, so these scores measure agreement "
    "with those rules, not confirmed cybersecurity accuracy."
)

# -----------------------------
# 11. Confusion Matrix
# -----------------------------
from sklearn.metrics import confusion_matrix

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    df["Known Unusual"],
    df["AI Prediction"],
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

col1, col2 = st.columns(2)

with col1:
    st.metric("True Negatives", int(tn))
    st.metric("False Positives", int(fp))

with col2:
    st.metric("False Negatives", int(fn))
    st.metric("True Positives", int(tp))

st.write("Confusion matrix (rows = known labels, columns = AI predictions):")

st.dataframe(
    pd.DataFrame(
        cm,
        index=["Known Normal", "Known Unusual"],
        columns=["AI Normal", "AI Suspicious"]
    )
)
