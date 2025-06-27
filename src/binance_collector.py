# binance_collector.py

import websocket
import json
import pandas as pd
from datetime import datetime
import os

SYMBOL = "ethusdt"
SAVE_PATH = "data"
os.makedirs(SAVE_PATH, exist_ok=True)

csv_file = os.path.join(SAVE_PATH, f"{SYMBOL}_ticks.csv")

def on_message(ws, message):
    data = json.loads(message)
    trade = {
        "timestamp": datetime.fromtimestamp(data["T"] / 1000.0),
        "price": float(data["p"]),
        "quantity": float(data["q"]),
        "side": "buy" if data["m"] is False else "sell"
    }

    print(trade)  # Affiche le tick en live

    # Append au CSV
    df = pd.DataFrame([trade])
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print(f"Subscribed to {SYMBOL}")
    payload = {
        "method": "SUBSCRIBE",
        "params": [f"{SYMBOL}@trade"],
        "id": 1
    }
    ws.send(json.dumps(payload))

if __name__ == "__main__":
    url = "wss://stream.binance.com:9443/ws"
    ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()
