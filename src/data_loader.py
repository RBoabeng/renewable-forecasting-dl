def load_and_preprocess(self):
    df = pd.read_csv(self.config['data']['raw_path'])
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # --- STEP 1: CREATE the columns first ---
    if 'hour_of_day' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
    if 'day_of_week' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # --- STEP 2: Now select them ---
    features = self.config['data']['features']
    target = self.config['data']['target_column']
    
    # Standardizing feature selection
    selected_cols = features + [target]
    
    try:
        data_scaled = self.scaler.fit_transform(df[selected_cols])
    except KeyError as e:
        print(f"Still missing a column: {e}")
        print(f"Available columns: {df.columns.tolist()}")
        raise
        
    return data_scaled