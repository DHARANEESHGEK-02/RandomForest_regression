import streamlit as st
import numpy as np
import pandas as pd  # Only for display/upload
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.set_page_config(layout="wide", page_icon="🏠")

st.title("🏠 Random Forest House Price Predictor")
st.markdown("**📁 Upload CSV or use sample • Live prediction • Cloud safe**")

# File upload
uploaded_file = st.file_uploader("📁 Upload CSV (Size,Bedrooms,Bathrooms,Age,Price)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df)} houses")

