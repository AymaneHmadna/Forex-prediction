import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["HADOOP_HOME"] = "C:\\hadoop"
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
SEQ_LENGTH = 60
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i   = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2*(i//2))/np.float32(d_model))
    angle_rads  = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)
inp  = layers.Input(shape=(SEQ_LENGTH, 1))
x    = layers.Dense(64)(inp)
x    = x + positional_encoding(SEQ_LENGTH, 64)
attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
x    = layers.Add()([x, attn])
x    = layers.LayerNormalization()(x)
x    = layers.Dropout(0.2)(x)
x    = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
x    = layers.Dropout(0.2)(x)
x    = layers.Bidirectional(layers.GRU(32))(x)
x    = layers.Dropout(0.2)(x)
x    = layers.Dense(64, activation="relu")(x)
x    = layers.Dropout(0.2)(x)
x    = layers.Dense(16, activation="relu")(x)
out  = layers.Dense(1)(x)
model = models.Model(inp, out)
model.load_weights('model_export/forex_weights.weights.h5')
scaler = joblib.load('model_export/scaler.pkl')
data   = yf.download("EURUSD=X", period="7d", interval="1m", progress=False)
prices = data[['Close']].dropna().values.flatten()
deltas = np.diff(prices)
scaled = scaler.transform(deltas.reshape(-1, 1))
X_test, y_true, prices_t = [], [], []
for i in range(len(scaled) - SEQ_LENGTH):
    X_test.append(scaled[i:i+SEQ_LENGTH])
    y_true.append(deltas[i+SEQ_LENGTH])
    prices_t.append(prices[i+SEQ_LENGTH])
X_test   = np.array(X_test)
y_true   = np.array(y_true)
prices_t = np.array(prices_t)
y_pred_delta = scaler.inverse_transform(
    model.predict(X_test, batch_size=64, verbose=0)
).flatten()
y_true_price = prices_t + y_true
y_pred_price = prices_t + y_pred_delta
errors_pips  = (y_pred_price - y_true_price) * 10000
mae   = mean_absolute_error(y_true_price, y_pred_price)
rmse  = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
mape  = np.mean(np.abs((y_true_price - y_pred_price) / y_true_price)) * 100
r2    = r2_score(y_true_price, y_pred_price)
dir_acc  = np.mean(np.sign(y_true) == np.sign(y_pred_delta)) * 100
hit_1    = np.mean(np.abs(errors_pips) <= 1)  * 100
hit_3    = np.mean(np.abs(errors_pips) <= 3)  * 100
hit_5    = np.mean(np.abs(errors_pips) <= 5)  * 100
hit_10   = np.mean(np.abs(errors_pips) <= 10) * 100
med_err  = np.median(np.abs(errors_pips))
max_err  = np.max(np.abs(errors_pips))
std_err  = np.std(errors_pips)
bias     = np.mean(errors_pips)
print(f"Test sequences             : {len(X_test)}")
print(f"MAE                        : {mae:.6f}  ({mae*10000:.2f} pips)")
print(f"RMSE                       : {rmse:.6f}  ({rmse*10000:.2f} pips)")
print(f"MAPE                       : {mape:.4f} %")
print(f"R2                         : {r2:.4f}")
print(f"Direction Accuracy         : {dir_acc:.2f} %")
print(f"Hit Rate <= 1 pip          : {hit_1:.2f} %")
print(f"Hit Rate <= 3 pips         : {hit_3:.2f} %")
print(f"Hit Rate <= 5 pips         : {hit_5:.2f} %")
print(f"Hit Rate <= 10 pips        : {hit_10:.2f} %")
print(f"Median error               : {med_err:.2f} pips")
print(f"Max error                  : {max_err:.2f} pips")
print(f"Standard deviation error   : {std_err:.2f} pips")
print(f"Average bias               : {bias:.2f} pips")
pd.DataFrame([{
    'MAE_price': round(mae,6), 'MAE_pips': round(mae*10000,2),
    'RMSE_price': round(rmse,6), 'RMSE_pips': round(rmse*10000,2),
    'MAPE_pct': round(mape,4), 'R2': round(r2,4),
    'Direction_Acc_pct': round(dir_acc,2),
    'Hit_1pip_pct': round(hit_1,2), 'Hit_5pips_pct': round(hit_5,2),
    'Median_pips': round(med_err,2), 'Max_pips': round(max_err,2),
    'Bias_pips': round(bias,2),
}]).to_csv('model_export/evaluation_results.csv', index=False)
