import pandas as pd
from supabase import create_client, Client
import streamlit as st

# Load secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# Create client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load your local training logs CSV
df = pd.read_csv("training_logs.csv")

# Convert DataFrame to list of dicts
records = df.to_dict(orient="records")

# Push each record to Supabase
for record in records:
    response = supabase.table("training_logs").insert(record).execute()
    print(response)
