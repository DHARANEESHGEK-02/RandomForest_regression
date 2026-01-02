import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

st.title("🏠 House Price Prediction with Random Forest Regression")

# File upload
uploaded_file = st.file_uploader("Upload CSV file (columns: Size, SqFt, Bedrooms, Bathrooms, Age, Price)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset preview:")
    st.dataframe(df.head())
    
    # Features and target
    feature_cols = ['Size', 'SqFt', 'Bedrooms', 'Bathrooms', 'Age']
    X = df[feature_cols]
    y = df['Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Model Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(importance, x='importance', y='feature', orientation='h',
                 title='Feature Importance')
    st.plotly_chart(fig)
    
    # Prediction interface
    st.subheader("Predict New House Price")
    col1, col2, col3, col4, col5 = st.columns(5)
    size = col1.number_input("Size (sqft)", value=2000)
    sq_ft = col2.number_input("SqFt", value=2000)
    bedrooms = col3.number_input("Bedrooms", value=3)
    bathrooms = col4.number_input("Bathrooms", value=2.0, step=0.5)
    age = col5.number_input("Age (years)", value=10)
    
    input_data = np.array([[size, sq_ft, bedrooms, bathrooms, age]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"Predicted Price: ${prediction:,.2f}")
    
else:
    st.info("👈 Please upload a CSV file to get started!")
