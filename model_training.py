import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

# Load processed data
train_df = pd.read_csv('processed_train.csv')
test_df = pd.read_csv('processed_test.csv')

# Healthy data: Assume RUL > 50 is healthy
healthy_train = train_df[train_df['rul'] > 50]
features = [col for col in train_df.columns if 'sensor' in col or 'efficiency' in col or 'vibration_rms' in col]

# Model 1: Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(healthy_train[features])
train_df['iso_anomaly'] = iso_forest.predict(train_df[features])
test_df['iso_anomaly'] = iso_forest.predict(test_df[features])

# Model 2: Autoencoder
input_dim = len(features)
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train on healthy data
autoencoder.fit(healthy_train[features], healthy_train[features], epochs=50, batch_size=32, verbose=0)

# Reconstruction error
train_recon = autoencoder.predict(train_df[features])
test_recon = autoencoder.predict(test_df[features])
train_df['recon_error'] = np.mean((train_df[features].values - train_recon)**2, axis=1)
test_df['recon_error'] = np.mean((test_df[features].values - test_recon)**2, axis=1)

# Threshold: 3 std above mean of healthy
threshold = train_df[train_df['rul'] > 50]['recon_error'].mean() + 3 * train_df[train_df['rul'] > 50]['recon_error'].std()
train_df['ae_anomaly'] = (train_df['recon_error'] > threshold).astype(int)
test_df['ae_anomaly'] = (test_df['recon_error'] > threshold).astype(int)

# Evaluation
true_labels = (test_df['rul'] <= 10).astype(int)  # Faulty if RUL <= 10
f1_iso = f1_score(true_labels, test_df['iso_anomaly'])
f1_ae = f1_score(true_labels, test_df['ae_anomaly'])
print(f"Isolation Forest F1: {f1_iso:.2f}, Autoencoder F1: {f1_ae:.2f}")

# Save models and data
autoencoder.save('autoencoder_model.h5')
train_df.to_csv('train_with_anomalies.csv', index=False)
test_df.to_csv('test_with_anomalies.csv', index=False)

# Plot example
plt.figure(figsize=(10, 5))
plt.plot(test_df['cycle'][:100], test_df['recon_error'][:100], label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.savefig('anomaly_plot.png')
plt.show()
