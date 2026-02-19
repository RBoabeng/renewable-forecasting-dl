import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

class MicrogridDataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler()
        
    def load_and_preprocess(self):
        df = pd.read_csv(self.config['data']['raw_path'])
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # --- STEP 1: CREATE the cyclical columns first ---
        if 'hour_of_day' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            
        if 'day_of_week' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # --- STEP 2: Select features based on config ---
        features = self.config['data']['features']
        target = self.config['data']['target_column']
        
        selected_cols = features + [target]
        
        try:
            data_scaled = self.scaler.fit_transform(df[selected_cols])
        except KeyError as e:
            print(f"ERROR: Missing column: {e}")
            print(f"Available columns: {df.columns.tolist()}")
            raise
            
        return data_scaled

    def create_sequences(self, data):
        lookback = self.config['model']['input_window']
        X, y = [], []
        
        for i in range(len(data) - lookback):
            X.append(data[i : i + lookback, :-1]) # All features except target
            y.append(data[i + lookback, -1])      # The target value
            
        return np.array(X), np.array(y)