import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Load CMAPSS dataset (FD001: Turbofan engine)
def load_cmapss_data(train_file='train_FD001.txt', test_file='test_FD001.txt'):
    if os.path.exists(train_file):
        train_df = pd.read_csv(train_file, sep=' ', header=None)
        test_df = pd.read_csv(test_file, sep=' ', header=None)
        # Column names (based on CMAPSS description)
        cols = ['unit', 'cycle'] + [f'sensor_{i}' for i in range(1, 22)] + ['rul']
        train_df.columns = cols[:len(train_df.columns)]
        test_df.columns = cols[:len(test_df.columns)]
        return train_df, test_df
    else:
        # Simulate data if dataset not available
        np.random.seed(42)
        cycles = 200
        units = 10
        data = []
        for unit in range(1, units + 1):
            for cycle in range(1, cycles + 1):
                row = [unit, cycle] + list(np.random.normal(0, 1, 21)) + [cycles - cycle]
                data.append(row)
        df = pd.DataFrame(data, columns=['unit', 'cycle'] + [f'sensor_{i}' for i in range(1, 22)] + ['rul'])
        return df, df  # Use same for train/test

def preprocess_data(df):
    # Sensor fusion: Add derived features
    df['efficiency'] = df['sensor_7'] / (df['sensor_3'] + 1e-6)  # Power / RPM proxy
    df['vibration_rms'] = np.sqrt(df[['sensor_4', 'sensor_5']].mean(axis=1))  # RMS vibration
    
    # Filtering: Moving average for noise reduction
    df = df.groupby('unit').apply(lambda x: x.rolling(window=5, min_periods=1).mean()).reset_index(drop=True)
    
    # Normalization
    scaler = MinMaxScaler()
    sensor_cols = [col for col in df.columns if 'sensor' in col or 'efficiency' in col or 'vibration_rms' in col]
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    
    # Segmentation: Group by operating state (simulate based on cycle)
    df['state'] = pd.cut(df['cycle'], bins=[0, 50, 100, 200], labels=['low_load', 'mid_load', 'high_load'])
    
    return df, scaler

if __name__ == "__main__":
    train_df, test_df = load_cmapss_data()
    train_processed, scaler = preprocess_data(train_df)
    test_processed, _ = preprocess_data(test_df)
    train_processed.to_csv('processed_train.csv', index=False)
    test_processed.to_csv('processed_test.csv', index=False)
    print("Preprocessing complete. Files saved.")
