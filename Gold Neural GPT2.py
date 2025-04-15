import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Configuration ---
SYMBOLS = ["XAUUSD", "EURUSD"]
TIMEFRAME = mt5.TIMEFRAME_H1
LOOKBACK_DAYS = 90
RETRAIN_INTERVAL = 7
MAX_DRAWDOWN = 0.15
MIN_BARS = 60
PROB_THRESHOLD = 0.05
NEWS_STOP_MINUTES = 1  # Stop trading 1 minute before news
NEWS_ORDER_WINDOW = 60  # Seconds before news to place orders

# --- MT5 Initialization ---
def init_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed.")
        sys.exit(1)
    
    login = 240315705
    password = "98777428Kk."
    server = "Exness-MT5Trial6"

    if not mt5.login(login=login, password=password, server=server):
        print(f"MT5 login failed: {mt5.last_error()}")
        sys.exit(1)
    print("MT5 initialized and logged in successfully")
    
    account_info = mt5.account_info()
    if account_info is None:
        print(f"Failed to retrieve account info: {mt5.last_error()}")
        sys.exit(1)
    print(f"Initial account balance: {account_info.balance}")

# --- Fetch High Impact News from Myfxbook ---
def fetch_high_impact_news():
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(hours=1)
    
    try:
        url = "https://www.myfxbook.com/forex-economic-calendar"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, params={"timeframe": "today"})
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        high_impact_news = []
        
        # Find the calendar table
        table = soup.find("table", {"id": "economicCalendarTable"})
        if not table:
            print("Could not find calendar table on Myfxbook.")
            return []
        
        # Parse rows
        rows = table.find_all("tr", {"class": "calendar_row"})
        for row in rows:
            # Check impact (High impact has class 'sentiment high')
            impact_cell = row.find("td", {"class": "sentiment"})
            if impact_cell and "high" in impact_cell.get("class", []):
                # Currency
                currency = row.find("td", {"class": "currency"}).text.strip()
                if currency not in ["USD", "EUR"]:
                    continue
                
                # Event time
                time_str = row.find("td", {"class": "time"}).text.strip()
                date_str = row.find("td", {"class": "date"}).text.strip()
                # Combine date and time (assume today’s date for simplicity)
                event_time_str = f"{date_str} {time_str}"
                try:
                    event_time = pd.to_datetime(event_time_str, format="%b %d %I:%M%p")
                    # Adjust year to current year
                    event_time = event_time.replace(year=utc_to.year)
                    # Convert to UTC (Myfxbook times are typically in user’s timezone, assume UTC for now)
                    if event_time < utc_from or event_time > utc_to:
                        continue
                except ValueError:
                    print(f"Failed to parse time: {event_time_str}")
                    continue
                
                # Event name
                event_name = row.find("td", {"class": "event"}).text.strip()
                
                # Previous and forecast
                previous = row.find("td", {"class": "previous"}).text.strip()
                forecast = row.find("td", {"class": "forecast"}).text.strip()
                
                try:
                    previous = float(previous.replace("%", "").replace("K", "e3") or 0)
                    forecast = float(forecast.replace("%", "").replace("K", "e3") or 0)
                except ValueError:
                    previous = 0
                    forecast = 0
                
                high_impact_news.append({
                    "time": event_time,
                    "currency": currency,
                    "name": event_name,
                    "previous": previous,
                    "forecast": forecast
                })
        
        print(f"High impact news: {len(high_impact_news)} events")
        return high_impact_news
    except Exception as e:
        print(f"Error fetching news from Myfxbook: {e}")
        print("Continuing without news data. Check network or Myfxbook access.")
        return []

# --- Predict News Impact ---
def predict_news_impact(news_event, symbol):
    forecast = news_event.get("forecast", 0)
    previous = news_event.get("previous", 0)
    
    currency = news_event["currency"]
    symbol_currencies = symbol[:3], symbol[3:]
    impact = 0  # Neutral
    
    if currency in symbol_currencies:
        if forecast > previous:
            impact = 1 if currency == symbol_currencies[0] else -1
        elif forecast < previous:
            impact = -1 if currency == symbol_currencies[0] else 1
    return impact  # 1: Bullish, -1: Bearish, 0: Neutral

# --- Data Pipeline ---
def fetch_historical_data(symbol, days=LOOKBACK_DAYS):
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}.")
        return pd.DataFrame()

    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, TIMEFRAME, utc_from, utc_to)
    
    if rates is None or len(rates) == 0:
        print(f"No historical data for {symbol}.")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['ctime'] = pd.to_datetime(df['time'], unit='s')
    df = df[['ctime', 'open', 'high', 'low', 'close', 'tick_volume']]
    return df

# --- Neural Network Model (LSTM) ---
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_data(df):
    if df.empty or len(df) < MIN_BARS:
        print(f"DataFrame has {len(df)} bars, less than required {MIN_BARS}.")
        return np.array([]), np.array([]), None
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['close']].values)
    X, y = [], []
    for i in range(MIN_BARS, len(scaled_data)):
        X.append(scaled_data[i-MIN_BARS:i])
        if scaled_data[i] > scaled_data[i-1]:
            y.append([1, 0, 0])
        elif scaled_data[i] < scaled_data[i-1]:
            y.append([0, 1, 0])
        else:
            y.append([0, 0, 1])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# --- LLM Decision Engine ---
def load_llm():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def llm_decision(signal_probs, news_impact=None, current_price=None, sl=None, tp=None, position=None):
    model, tokenizer = load_llm()
    probs_str = f"[Up: {signal_probs[0]:.3f} | Down: {signal_probs[1]:.3f} | Side: {signal_probs[2]:.3f}]"
    direction = "Up" if signal_probs[0] == max(signal_probs) else "Down" if signal_probs[1] == max(signal_probs) else "Sideways"

    if position is None:
        up_prob, down_prob = signal_probs[0], signal_probs[1]
        if news_impact is not None:
            decision = "BUY" if news_impact > 0 else "SELL" if news_impact < 0 else "HOLD"
            reasoning_raw = f"News impact predicted: {'Bullish' if news_impact > 0 else 'Bearish' if news_impact < 0 else 'Neutral'}."
        else:
            if up_prob > down_prob + PROB_THRESHOLD:
                decision = "BUY"
                reasoning_raw = "Strong upward trend predicted."
            elif down_prob > up_prob + PROB_THRESHOLD:
                decision = "SELL"
                reasoning_raw = "Strong downward trend predicted."
            else:
                decision = "HOLD"
                reasoning_raw = "Trend unclear."
        reasoning = f"📈 Predict: {probs_str} ({direction}) ✅ Decision: {decision} 🧠 Reason: {reasoning_raw}"
    else:
        position_direction = "Up" if position.type == mt5.ORDER_TYPE_BUY else "Down"
        if current_price >= tp and position.type == mt5.ORDER_TYPE_BUY:
            decision = "TAKE PROFIT"
            reasoning_raw = "Price reached TP."
        elif current_price <= tp and position.type == mt5.ORDER_TYPE_SELL:
            decision = "TAKE PROFIT"
            reasoning_raw = "Price reached TP."
        elif direction == position_direction:
            decision = "HOLD"
            reasoning_raw = "Trend aligns with position."
        else:
            decision = "CUT LOSS"
            reasoning_raw = "Trend reversed."
        reasoning = f"📈 Predict: {probs_str} ({direction}) ✅ Decision: {decision} 🧠 Reason: {reasoning_raw}"

    confidence = max(signal_probs) * 100
    return decision, reasoning, confidence

# --- Risk Management ---
def calculate_atr(high, low, close, period=14):
    tr = np.maximum(high[1:] - low[1:], 
                    np.abs(high[1:] - close[:-1]), 
                    np.abs(low[1:] - close[:-1]))
    atr = np.zeros(len(high))
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
    return atr

def calculate_atr_stop_loss(df, period=14):
    if df.empty:
        print("DataFrame is empty. Cannot calculate ATR.")
        return 0
    atr = calculate_atr(df['high'].values, df['low'].values, df['close'].values, period)
    return atr[-1] * 2

def calculate_position_size(account_balance, symbol, risk_per_trade=0.01):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 0.1
    min_volume = symbol_info.volume_min
    risk_amount = account_balance * risk_per_trade
    volume = max(min_volume, risk_amount / 1000)
    volume = round(volume / min_volume) * min_volume
    return min(volume, symbol_info.volume_max)

# --- Check Open Positions ---
def get_open_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return None
    return positions[0]

# --- Close Position ---
def close_position(position):
    symbol = position.symbol
    ticket = position.ticket
    volume = position.volume
    position_type = position.type

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get tick data for {symbol}")
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL if position_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": ticket,
        "price": tick.bid if position_type == mt5.ORDER_TYPE_BUY else tick.ask,
        "comment": "AI Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to close position {ticket}: {result.comment}")
    else:
        print(f"Position {ticket} closed successfully.")
    return result

# --- Place Stop/Limit Order ---
def place_pending_order(symbol, order_type, price, volume, sl, tp):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "comment": "AI News Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Pending order failed for {symbol}: {result.comment}")
    else:
        print(f"Pending order placed: {order_type} at {price}")
    return result

# --- Trade Execution ---
def send_trade(symbol, signal, tp, sl, volume):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    if signal == "BUY":
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "sl": sl,
            "tp": tp,
            "comment": "AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
    elif signal == "SELL":
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL,
            "price": tick.bid,
            "sl": sl,
            "tp": tp,
            "comment": "AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
    else:
        return None
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Trade failed for {symbol}: {result.comment}")
    else:
        print(f"Trade executed: {signal} {volume} lots")
    return result

# --- Self-Learning Mechanism ---
def retrain_model(model, X, y):
    if len(X) == 0 or len(y) == 0:
        print("No data to retrain model.")
        return model
    model.fit(X, y, epochs=10, batch_size=64, verbose=0)
    return model

def monitor_performance(account_balance, initial_balance):
    drawdown = (initial_balance - account_balance) / initial_balance
    if drawdown > MAX_DRAWDOWN:
        print("Max drawdown exceeded. Stopping system.")
        return False
    return True

# --- Main Trading Loop ---
def main():
    init_mt5()
    last_retrain = datetime.now() - timedelta(days=RETRAIN_INTERVAL + 1)
    account_info = mt5.account_info()
    if account_info is None:
        print(f"Failed to get account info: {mt5.last_error()}")
        sys.exit(1)
    initial_balance = account_info.balance

    default_model = build_lstm_model((MIN_BARS, 1))

    while True:
        current_time = datetime.now()
        if (current_time - last_retrain).days >= RETRAIN_INTERVAL:
            print("Retraining model...")
            for symbol in SYMBOLS:
                df = fetch_historical_data(symbol)
                X, y, scaler = preprocess_data(df)
                if len(X) > 0:
                    model = build_lstm_model((X.shape[1], X.shape[2]))
                    model = retrain_model(model, X, y)
                else:
                    model = default_model
            last_retrain = current_time

        # Fetch High Impact News
        high_impact_news = fetch_high_impact_news()
        trading_allowed = True
        news_imminent = False
        news_event = None

        # Check if news is approaching
        for event in high_impact_news:
            time_to_news = (event["time"] - current_time).total_seconds()
            if 0 < time_to_news <= NEWS_STOP_MINUTES * 60:
                trading_allowed = False
                print(f"High impact news in {time_to_news:.0f} seconds: {event['name']}. Pausing new trades.")
                if NEWS_ORDER_WINDOW - 10 <= time_to_news <= NEWS_ORDER_WINDOW + 10:
                    news_imminent = True
                    news_event = event
                break

        for symbol in SYMBOLS:
            df = fetch_historical_data(symbol)
            X, _, scaler = preprocess_data(df)
            if len(X) == 0:
                print(f"No valid data for {symbol}. Skipping.")
                continue
            signal_probs = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]), verbose=0)[0]

            position = get_open_position(symbol)
            if position is None:
                if not trading_allowed:
                    print(f"Trading paused for {symbol} due to upcoming news.")
                    continue

                if news_imminent and news_event:
                    news_impact = predict_news_impact(news_event, symbol)
                    decision, reasoning, confidence = llm_decision(signal_probs, news_impact=news_impact)
                    
                    atr_sl = calculate_atr_stop_loss(df)
                    current_price = df['close'].iloc[-1]
                    account_info = mt5.account_info()
                    volume = calculate_position_size(account_info.balance, symbol)

                    if decision in ["BUY", "SELL"]:
                        order_price = current_price + atr_sl * 0.5 if decision == "BUY" else current_price - atr_sl * 0.5
                        tp = order_price + atr_sl * 1.5 if decision == "BUY" else order_price - atr_sl * 1.5
                        sl = order_price - atr_sl if decision == "BUY" else order_price + atr_sl
                        order_type = mt5.ORDER_TYPE_BUY_STOP if decision == "BUY" else mt5.ORDER_TYPE_SELL_STOP
                        
                        trade_signal = {
                            "pair": symbol,
                            "signal": f"Pending {decision}",
                            "confidence": float(confidence),
                            "reason": reasoning,
                            "order_price": float(order_price),
                            "tp": float(tp),
                            "sl": float(sl),
                            "volume": float(volume)
                        }
                        print(json.dumps(trade_signal, indent=2))
                        place_pending_order(symbol, order_type, order_price, volume, sl, tp)
                    else:
                        print(f"No trade for {symbol}: {reasoning}")
                else:
                    decision, reasoning, confidence = llm_decision(signal_probs)
                    atr_sl = calculate_atr_stop_loss(df)
                    current_price = df['close'].iloc[-1]
                    tp = current_price + atr_sl * 1.5 if decision == "BUY" else current_price - atr_sl * 1.5
                    sl = current_price - atr_sl if decision == "BUY" else current_price + atr_sl
                    account_info = mt5.account_info()
                    volume = calculate_position_size(account_info.balance, symbol)

                    trade_signal = {
                        "pair": symbol,
                        "signal": decision,
                        "confidence": float(confidence),
                        "reason": reasoning,
                        "tp": float(tp),
                        "sl": float(sl),
                        "volume": float(volume)
                    }
                    print(json.dumps(trade_signal, indent=2))
                    if decision in ["BUY", "SELL"]:
                        send_trade(symbol, decision, tp, sl, volume)
            else:
                current_price = mt5.symbol_info_tick(symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
                sl = position.sl
                tp = position.tp
                decision, reasoning, confidence = llm_decision(signal_probs, current_price=current_price, sl=sl, tp=tp, position=position)
                
                position_signal = {
                    "pair": symbol,
                    "current_price": float(current_price),
                    "sl": float(sl),
                    "tp": float(tp),
                    "decision": decision,
                    "confidence": float(confidence),
                    "reason": reasoning
                }
                print(json.dumps(position_signal, indent=2))
                
                if decision in ["CUT LOSS", "TAKE PROFIT"]:
                    close_position(position)

            if not monitor_performance(account_info.balance, initial_balance):
                print("System stopped due to max drawdown.")
                return

        time.sleep(10)

if __name__ == "__main__":
    main()
