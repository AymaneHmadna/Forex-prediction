import os
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["PATH"] + ";C:\\hadoop\\bin"
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import yfinance as yf
from collections import deque
from kafka import KafkaConsumer
import json
seq_length = 60
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
model.load_weights('model_export/forex_weights.weights.h5')
scaler = joblib.load('model_export/scaler.pkl')
price_memory = deque(maxlen=seq_length + 1)
historical = yf.download(tickers='EURUSD=X', period='2d', interval='1m', progress=False)
if not historical.empty:
    last_prices = historical['Close'].dropna().values[-(seq_length + 1):].flatten()
    for price in last_prices:
        price_memory.append(float(price))
consumer = KafkaConsumer(
    'forex_live',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    enable_auto_commit=True
)
for message in consumer:
    tick = message.value
    current_price = tick['close_price']
    timestamp = tick['timestamp']
    price_memory.append(current_price)
    if len(price_memory) < seq_length + 1:
        print(f"[{timestamp}] received price : {current_price:.5f} | memory loading: {len(price_memory)}/{seq_length+1}...")
    else:
        prices_array = np.array(price_memory)
        deltas = np.diff(prices_array) 
        scaled_deltas = scaler.transform(deltas.reshape(-1, 1))
        X_live = scaled_deltas.reshape(1, seq_length, 1)
        scaled_pred_delta = model.predict(X_live, verbose=0)
        real_delta = scaler.inverse_transform(scaled_pred_delta)[0][0]
        real_pred = current_price + real_delta
        delta_pips = real_delta * 10000
        print(f"[{timestamp}] price of this min: {current_price:.5f}\Predicted price for next min: {real_pred:.5f} ({delta_pips:+.1f} pips)")
