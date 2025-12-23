import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

# Page Configuration
st.set_page_config(
    page_title="Corn Yield Predictor",
    page_icon="🌽",
    layout="wide"
)

# Load Model
model_filename = 'best_corn_xgboost.pkl'

@st.cache_resource
def load_model(filename):
    try:
        # Load the XGBoost Model
        model = joblib.load(filename=filename)
        print(" Model loaded successfully.")
        return model
    except FileNotFoundError:
        return None
    
model = load_model(model_filename)

# Sidebar Inputs
st.sidebar.header("🎛️ Field Parameters")
st.sidebar.markdown("Adjust the conditions to predict yield.")

def user_input_features():
    # Weather Parameters
    st.sidebar.subheader("🌤️ Weather")

    # Constraining the inputs ranges based on my clead data
    max_temp = st.sidebar.slider('Max Temperature (°C)', 20.0, 40.0, 30.0)
    min_temp = st.sidebar.slider('Min Temperature (°C)', 10.0, 30.0, 22.0)

    # Auto calculate Avg_Temp
    avg_temp = (max_temp + min_temp) / 2

    precip = st.sidebar.slider('Rainfall (mm)', 0.0, 300.0, 150.0)
    wind = st.sidebar.slider('Wind Speed (m/s)', 0.0, 10.0, 2.5)

    # Soil Controls
    st.sidebar.subheader("🌱 Soil Composition")
    pH = st.sidebar.slider('Soil pH', 4.0, 9.0, 6.5)
    
    # Soil Texture (Must sum to ~100%, but we let the user slide freely for now)
    clay = st.sidebar.slider('Clay %', 0, 100, 30)
    sand = st.sidebar.slider('Sand %', 0, 100, 40)
    silt = st.sidebar.slider('Silt %', 0, 100, 30)

    # Create the DataFrame with EXACT column names from training
    data = {
        'Avg_Temp': avg_temp,
        'Min_Temp': min_temp,
        'Max_Temp': max_temp,
        'Avg_Precipitation': precip,
        'Wind_Speed': wind,
        'pH': pH,
        'Clay': clay,
        'Sand': sand,
        'Silt': silt
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main Dashboard
st.title("🌽 Corn Yield Prediction System")
st.markdown("Powered by **XGBoost** | Optimized for **Physics-Based** Forecasting")

col1, col2 = st.columns([1, 2])

with col1:
    # Display User Inputs
    st.subheader("Current Field Status")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}))

    # Soil Check
    total_soil = input_df['Clay'][0] + input_df['Sand'][0] + input_df['Silt'][0]
    if total_soil != 100:
        st.error(f"Soil Composition Warning: Total = {total_soil}%. Should sum to 100%.")

with col2:
    # Display Prediction
    st.subheader(" Model Forecast")

    if model is None:
        st.error(" Error: 'best_corn_xgboost.pkl' model file not found.")
    else:
        # Predict
        prediction = model.predict(input_df)[0]

        # Color Logic
        if prediction < 1.5:
            color = 'red'
            status = 'Low Yield'
        elif prediction < 2.2:
            color = 'orange'
            status = 'Average Yield'
        else:
            color = 'green'
            status = 'High Yield'

        st.markdown(f"""
        <div style="text-align: center; border: 2px solid #ddd; padding: 20px; border-radius: 10px;">
            <h2 style="color: grey; margin:0;">Estimated Efficiency</h2>
            <h1 style="color: {color}; font-size: 60px; margin:0;">{prediction:.2f} t/ha</h1>
            <h3 style="color: {color};">{status}</h3>
        </div>
        """, unsafe_allow_html=True)

# This ignores all warnings (including the thread context ones)
warnings.filterwarnings("ignore")