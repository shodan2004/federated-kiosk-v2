# streamlit_app.py

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import base64
import pyttsx3  # 🔊 Voice alerts

# 🌐 Connect to Supabase PostgreSQL using SQLAlchemy
SUPABASE_DB_URL = "postgresql://postgres.xjvteeaqbttimgcyjzyu:shodhanAdmin123@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
engine = create_engine(SUPABASE_DB_URL)

# 🔄 Load data function
def load_data():
    df = pd.read_sql("SELECT * FROM training_logs ORDER BY timestamp ASC", engine)
    return df

# 🔊 Voice alert function
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# 🚨 Accuracy drop checker

def check_accuracy_alert(df):
    if 'accuracy' not in df.columns:
        return
    latest_round = df['round'].max()
    recent = df[df['round'] == latest_round]
    if recent['accuracy'].mean() < 0.75:
        st.error("⚠️ Model accuracy has dropped below 75%!")
        speak("Warning! Accuracy has dropped below 75 percent")

# 📊 Main dashboard
st.set_page_config(page_title="Federated Kiosk Dashboard", layout="wide")
st.title("📟 Federated Learning Monitoring Dashboard")

# Load data
df = load_data()
check_accuracy_alert(df)

# 🔎 Sidebar filters
st.sidebar.header("🔎 Filters")
rounds = sorted(df['round'].unique())
clients = sorted(df['client_id'].unique())

selected_round = st.sidebar.selectbox("Select Training Round", rounds)
selected_client = st.sidebar.selectbox("Select Client ID", clients)

# 🎯 Filtered Data
filtered_df = df[(df['round'] == selected_round) & (df['client_id'] == selected_client)]
st.subheader(f"Client {selected_client} | Round {selected_round} Log")
st.dataframe(filtered_df)

# 📈 Metrics
st.subheader("📊 Training Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("📉 Loss", f"{filtered_df['loss'].values[-1]:.4f}")
col2.metric("📉 Validation Loss", f"{filtered_df['val_loss'].values[-1]:.4f}")
if 'accuracy' in filtered_df.columns:
    col3.metric("✅ Accuracy", f"{filtered_df['accuracy'].values[-1]:.4f}")
else:
    col3.write("Accuracy data not available")

# 🖼️ Round-wise visualization
st.subheader("📈 Round-wise Accuracy")
if 'accuracy' in df.columns:
    chart_data = df.groupby("round")["accuracy"].mean().reset_index()
    st.line_chart(chart_data, x="round", y="accuracy")

# 📥 Download logs
st.subheader("⬇️ Export Logs")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "training_logs.csv", "text/csv")
