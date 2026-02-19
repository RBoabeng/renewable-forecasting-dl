import torch
import yaml
import joblib
import numpy as np
import pandas as pd

from src.model import MicrogridLSTM

print("⚡ Initializing Microgrid Forecasting System...")

# 1. Load Configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Load the Scaler
scaler_path = "outputs/scalers/microgrid_scaler_v1.pkl"
scaler = joblib.load(scaler_path)
print("✅ Data Scaler loaded.")

# 3. Rebuild the Model Architecture and Load Weights
# FIX: Match the 7 features the model was originally trained on
input_features = len(config['data']['features']) 
model = MicrogridLSTM(
    input_size=input_features, 
    hidden_size=config['model']['hidden_size'], 
    num_layers=config['model']['num_layers'], 
    output_size=config['model']['output_window']
)

model_path = "outputs/models/microgrid_lstm_v1.pth"
# FIX: Add weights_only=True to resolve the PyTorch security warning
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval() 
print("✅ LSTM Model Weights loaded.")

# 4. Simulate Live Incoming Data
df = pd.read_csv(config['data']['raw_path'])
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

if 'hour_of_day' in df.columns:
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
if 'day_of_week' in df.columns:
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# The scaler expects all features + the target
selected_cols = config['data']['features'] + [config['data']['target_column']]
recent_data = df[selected_cols].tail(config['model']['input_window']) 

# 5. Preprocess and Predict
recent_data_scaled = scaler.transform(recent_data)

# FIX: Slice off the target column (:-1) so we only feed the 7 features into the model
X_live_data = recent_data_scaled[:, :-1]
X_live = torch.tensor(X_live_data, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    scaled_prediction = model(X_live).numpy()

# 6. Inverse Transform the Prediction back to kW
# The scaler was trained on (features + target), so dummy array needs to be that size
dummy_array = np.zeros((1, input_features + 1))
dummy_array[0, -1] = scaled_prediction[0, 0]
final_kw_prediction = scaler.inverse_transform(dummy_array)[0, -1]

print("\n" + "="*50)
print(f"🔮 PREDICTED ENERGY GENERATION FOR NEXT HOUR: {final_kw_prediction:.2f} kW")
print("="*50 + "\n")