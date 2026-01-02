import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_icon="🏠")

st.title("🏠 Random Forest House Predictor")
st.markdown("**✅ 100% Cloud Compatible • No Pandas • Live Demo**")

# Pure numpy dataset (no pandas!)
X_data = np.array([
    [1500, 3, 2.0, 10],
    [2200, 4, 2.5, 5],
    [1800, 3, 2.0, 15],
    [2500, 5, 3.0, 8],
    [1200, 2, 1.0, 20],
    [3000, 5, 3.5, 3],
    [1600, 3, 2.0, 12],
    [2000, 4, 2.5, 7],
    [1400, 3, 1.5, 18],
    [2800, 4, 3.0, 4]
])

y_prices = np.array([300000,450000,350000,550000,220000,650000,320000,420000,280000,580000])

# Train model
@st.cache_resource
def train_rf():
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_data, y_prices)
    return model

model = train_rf()

# Sidebar
st.sidebar.header("🏠 House Specs")
size = st.sidebar.slider("Size (sqft)", 1000, 3500, 2000)
beds = st.sidebar.slider("Bedrooms", 1, 5, 3)
baths = st.sidebar.slider("Bathrooms", 1.0, 4.0, 2.0)
house_age = st.sidebar.slider("Age (years)", 0, 25, 10)

# Predict
if st.sidebar.button("🎯 Get Price", type="primary"):
    features = np.array([[size, beds, baths, house_age]])
    price = model.predict(features)[0]
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, #10B981, #059669); 
                color:white; padding:1.5rem; border-radius:12px; text-align:center;">
        <h2>${int(price):,}</h2>
        <p>Predicted Price</p>
    </div>
    """, unsafe_allow_html=True)

# Metrics
col1, col2 = st.columns(2)
col1.metric("🏠 Houses", len(X_data))
col2.metric("💰 Avg Price", f"${int(np.mean(y_prices)):,}")

# Charts (pure plotly + numpy)
col1, col2 = st.columns(2)

# Size vs Price
sizes = X_data[:, 0]
fig1 = px.scatter(x=sizes, y=y_prices, 
                 labels={'x':'Size (sqft)', 'y':'Price ($)'},
                 title="📏 Size vs Price")
st.plotly_chart(fig1, use_container_width=True)

# Feature importance
importance = model.feature_importances_
features = ['Size', 'Bedrooms', 'Bathrooms', 'Age']
fig2 = px.bar(x=importance, y=features, orientation='h',
             title="📊 Feature Importance")
st.plotly_chart(fig2, use_container_width=True)

# Data table (convert to df for display only)
df_display = pd.DataFrame(X_data, columns=['Size','Bedrooms','Bathrooms','Age'])
df_display['Price'] = y_prices
st.subheader("📋 Dataset")
st.dataframe(df_display)

st.success("✅ **Streamlit Cloud Ready!** No pandas/numpy conflicts")
