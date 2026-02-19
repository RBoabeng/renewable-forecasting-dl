import streamlit as st
import torch
import yaml
import joblib
import numpy as np
import pandas as pd

# Import your custom model class
from src.model import MicrogridLSTM

# --- 1. Page Setup ---
st.set_page_config(page_title="Microgrid AI", page_icon="⚡", layout="wide")
st.title("⚡ Renewable Energy Microgrid Forecaster")
st.markdown("This app uses a trained PyTorch LSTM to predict the next hour of renewable energy generation based on the last 24 hours of weather and grid data.")

# --- 2. Load Artifacts (Cached so it doesn't reload on every click) ---
@st.cache_resource
def load_system():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    scaler = joblib.load("outputs/scalers/microgrid_scaler_v1.pkl")
    input_features = len(config['data']['features']) 
    
    model = MicrogridLSTM(
        input_size=input_features, 
        hidden_size=config['model']['hidden_size'], 
        num_layers=config['model']['num_layers'], 
        output_size=config['model']['output_window']
    )
    model.load_state_dict(torch.load("outputs/models/microgrid_lstm_v1.pth", weights_only=True))
    model.eval()
    
    return config, scaler, model

config, scaler, model = load_system()

# --- 3. Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(config['data']['raw_path'])
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    if 'hour_of_day' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    if 'day_of_week' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
    return df

df = load_data()
selected_cols = config['data']['features'] + [config['data']['target_column']]
recent_data = df[selected_cols].tail(config['model']['input_window']).copy()

# --- 4. User Interface ---
st.subheader("📊 Last 24 Hours of Grid Data")
st.line_chart(recent_data[config['data']['target_column']])

st.subheader("🎛️ Adjust Current Weather Conditions")
st.markdown("Tweak the *latest* hour's weather below to see how the LSTM adapts its prediction!")

col1, col2, col3 = st.columns(3)
with col1:
    new_solar = st.slider("Solar Irradiance (W/m²)", 0.0, 1000.0, float(recent_data['solar_irradiance'].iloc[-1]))
with col2:
    new_wind = st.slider("Wind Speed (m/s)", 0.0, 25.0, float(recent_data['wind_speed'].iloc[-1]))
with col3:
    new_temp = st.slider("Temperature (°C)", -10.0, 40.0, float(recent_data['temperature'].iloc[-1]))

# Update the last row with the slider values
recent_data.iloc[-1, recent_data.columns.get_loc('solar_irradiance')] = new_solar
recent_data.iloc[-1, recent_data.columns.get_loc('wind_speed')] = new_wind
recent_data.iloc[-1, recent_data.columns.get_loc('temperature')] = new_temp

# --- 5. Prediction Logic ---
st.markdown("---")
if st.button("🔮 Predict Next Hour", type="primary", use_container_width=True):
    # Scale data
    recent_data_scaled = scaler.transform(recent_data)
    
    # Slice off target and create tensor
    X_live_data = recent_data_scaled[:, :-1]
    X_live = torch.tensor(X_live_data, dtype=torch.float32).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        scaled_prediction = model(X_live).numpy()
        
    # Inverse transform
    dummy_array = np.zeros((1, len(config['data']['features']) + 1))
    dummy_array[0, -1] = scaled_prediction[0, 0]
    final_kw = scaler.inverse_transform(dummy_array)[0, -1]
    
    st.success(f"### ⚡ Predicted Energy Generation: {final_kw:.2f} kW")