# streamlit_app.py

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# 🚫 Removed pyttsx3 to prevent ModuleNotFoundError on Streamlit Cloud

# 🔐 Load Supabase DB URL from Streamlit secrets
SUPABASE_DB_URL = "postgresql://postgres.xjvteeaqbttimgcyjzyu:shodhanAdmin123@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
engine = create_engine(SUPABASE_DB_URL)

# 🔄 Load data from Supabase
def load_data():
    df = pd.read_sql("SELECT * FROM training_logs ORDER BY timestamp ASC", engine)
    return df

# 🚨 Accuracy alert if val_accuracy drops below 75%
def check_accuracy_alert(df):
    if "val_accuracy" not in df.columns:
        return
    latest_round = df["round"].max()
    recent = df[df["round"] == latest_round]
    if recent["val_accuracy"].mean() < 0.75:
        st.error("⚠️ Model validation accuracy has dropped below 75%!")

# 🌐 Streamlit app layout
st.set_page_config(page_title="Federated Kiosk Dashboard", layout="wide")
st.title("📟 Federated Learning Monitoring Dashboard")

# Load data
df = load_data()
check_accuracy_alert(df)

# 🔎 Sidebar filters
st.sidebar.header("🔎 Filters")
rounds = sorted(df["round"].dropna().unique())
clients = sorted(df["client_id"].dropna().unique())

selected_round = st.sidebar.selectbox("Select Training Round", rounds)
selected_client = st.sidebar.selectbox("Select Client ID", clients)

# 🎯 Filtered Data
filtered_df = df[(df["round"] == selected_round) & (df["client_id"] == selected_client)]
st.subheader(f"Client {selected_client} | Round {selected_round} Log")
st.dataframe(filtered_df)

# 📊 Metrics Display
st.subheader("📊 Training Metrics")
col1, col2, col3 = st.columns(3)

# Safely handle missing columns
if not filtered_df.empty:
    if "train_loss" in filtered_df.columns:
        col1.metric("📉 Train Loss", f"{filtered_df['train_loss'].values[-1]:.4f}")
    else:
        col1.write("Train Loss not available")

    if "val_loss" in filtered_df.columns:
        col2.metric("📉 Validation Loss", f"{filtered_df['val_loss'].values[-1]:.4f}")
    else:
        col2.write("Validation Loss not available")

    if "val_accuracy" in filtered_df.columns:
        col3.metric("✅ Validation Accuracy", f"{filtered_df['val_accuracy'].values[-1]:.4f}")
    else:
        col3.write("Validation Accuracy not available")
else:
    st.warning("No data available for this round/client.")

# 📈 Round-wise Accuracy
st.subheader("📈 Round-wise Validation Accuracy")
if "val_accuracy" in df.columns:
    chart_data = df.groupby("round")["val_accuracy"].mean().reset_index()
    st.line_chart(chart_data, x="round", y="val_accuracy")
else:
    st.info("No validation accuracy data available.")

# 📥 Export Logs
st.subheader("⬇️ Export Logs")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "training_logs.csv", "text/csv")
