import yfinance as yf
import time
import json
from kafka import KafkaProducer
from datetime import datetime
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
while True:
        live_data = yf.download(tickers='EURUSD=X', period='1d', interval='1m', progress=False)
        if not live_data.empty:
            latest_row = live_data.iloc[-1]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_price = float(latest_row['Close'].iloc[0]) 
            market_tick = {
                'timestamp': current_time,
                'symbol': 'EURUSD',
                'close_price': current_price
            }
            producer.send('forex_live', value=market_tick)
            print(f"[{current_time}]EUR/USD price: {current_price:.5f}")
        time.sleep(60)
