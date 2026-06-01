import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import os
data = yf.download("EURUSD=X", start="2020-01-01", end="2025-01-01")
data = data[['Close']].dropna()
prices = data['Close'].values.flatten()
deltas = np.diff(prices) 
scaler = StandardScaler()
scaled_deltas = scaler.fit_transform(deltas.reshape(-1, 1))
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])     
        y.append(data[i+seq_length])         
    return np.array(X), np.array(y)
seq_length = 60
X, y = create_sequences(scaled_deltas, seq_length)
def positional_encoding(seq_len, d_model): 
    pos = np.arange(seq_len)[:, np.newaxis] 
    i = np.arange(d_model)[np.newaxis, :] 
    angle_rates = 1 / np.power(10000, (2*(i//2))/np.float32(d_model)) 
    angle_rads = pos * angle_rates 
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) 
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2]) 
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32) 
inputs = layers.Input(shape=(seq_length, 1)) 
x = layers.Dense(64)(inputs) 
x = x + positional_encoding(seq_length, 64) 
attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
x = layers.Add()([x, attn]) 
x = layers.LayerNormalization()(x) 
x = layers.Dropout(0.2)(x) 
x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x) 
x = layers.Dropout(0.2)(x)
x = layers.Bidirectional(layers.GRU(32))(x) 
x = layers.Dropout(0.2)(x) 
x = layers.Dense(64, activation="relu")(x) 
x = layers.Dropout(0.2)(x) 
x = layers.Dense(16, activation="relu")(x) 
outputs = layers.Dense(1)(x) 
model = models.Model(inputs, outputs)
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001) 
model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=["mae"]) 
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
history = model.fit(
    X, y,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)
if not os.path.exists('model_export'):
    os.makedirs('model_export')
model.save_weights('model_export/forex_weights.weights.h5')
joblib.dump(scaler, 'model_export/scaler.pkl')
import json
history_dict = {
    'loss':     [float(v) for v in history.history['loss']],
    'val_loss': [float(v) for v in history.history['val_loss']],
    'mae':      [float(v) for v in history.history['mae']],
    'val_mae':  [float(v) for v in history.history['val_mae']],
    'epochs':   len(history.history['loss']),
    'best_epoch': int(np.argmin(history.history['val_loss'])) + 1,
}
with open('model_export/training_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"best epoch : {history_dict['best_epoch']} / {history_dict['epochs']}")
print(f"best val_loss : {min(history_dict['val_loss']):.4f}")
print(f"best val_mae  : {min(history_dict['val_mae']):.4f}")
