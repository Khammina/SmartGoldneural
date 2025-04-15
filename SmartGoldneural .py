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
import logging
import telegram
import asyncio
import os
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Configuration ---
SYMBOLS = ["XAUUSD", "EURUSD"]
TIMEFRAME = mt5.TIMEFRAME_H1
LOOKBACK_DAYS = 90
RETRAIN_INTERVAL = 7
MAX_DRAWDOWN = 0.15|
MIN_BARS = 60
PROB_THRESHOLD = 0.05
RISK_PER_TRADE = 1
NEWS_STOP_MINUTES = 15
TELEGRAM_BOT_TOKEN = "7165263301:AAGAVwbK938E3WXuqpFQAl1P9RoWrAHm52s"
TELEGRAM_CHAT_ID = "6501082183"

# --- Logging Setup ---
logging.basicConfig(
    filename="ea_trading.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# --- Telegram Notification ---
async def send_telegram_message(message):
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

def notify(message):
    logger.info(message)
    success = asyncio.run(send_telegram_message(message))
    if not success:
        logger.warning("Telegram notification failed. Check bot token and chat ID.")

# --- MT5 Initialization ---
def init_mt5(max_retries=3):
    for attempt in range(max_retries):
        if mt5.initialize():
            login = 7002223
            password = "kH@N18Vb"
            server = "MohicansMarkets-Live"
            if mt5.login(login=login, password=password, server=server):
                logger.info("MT5 initialized and logged in successfully")
                account_info = mt5.account_info()
                if account_info:
                    logger.info(f"Initial account balance: {account_info.balance}")
                    return True
                logger.error("Failed to retrieve account info")
            else:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
        logger.warning(f"MT5 initialization failed, attempt {attempt+1}/{max_retries}")
        time.sleep(5)
    logger.error("MT5 initialization failed after max retries")
    return False

# --- Fetch High Impact News from Investing.com ---
def fetch_high_impact_news():
    try:
        url = "https://www.investing.com/economic-calendar/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        high_impact_news = []
        
        events = soup.find_all("tr", {"class": "js-event-item"})
        current_time = datetime.now()
        
        for event in events:
            impact = event.find("span", {"class": "icon"})
            if impact and "high" in impact.get("class", []):
                currency = event.find("td", {"class": "flagCur"}).text.strip().split()[0]
                if currency not in ["USD", "EUR"]:
                    continue
                
                time_str = event.find("td", {"class": "time"}).text.strip()
                try:
                    event_time = pd.to_datetime(f"{current_time.date()} {time_str}")
                    if event_time > current_time:
                        event_time = event_time.replace(year=current_time.year)
                    else:
                        event_time = event_time.replace(day=current_time.day + 1)
                except ValueError:
                    logger.warning(f"Failed to parse time: {time_str}")
                    continue
                
                event_name = event.find("td", {"class": "event"}).text.strip()
                high_impact_news.append({
                    "time": event_time,
                    "currency": currency,
                    "name": event_name
                })
        
        logger.info(f"High impact news: {len(high_impact_news)} events")
        return high_impact_news
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []

# --- Data Pipeline ---
def fetch_historical_data(symbol, days=LOOKBACK_DAYS, timeframe=TIMEFRAME):
    try:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select {symbol}")
            return pd.DataFrame()
        
        utc_to = datetime.now()
        utc_from = utc_to - timedelta(days=days)
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
        
        if rates is None or len(rates) == 0:
            logger.error(f"No historical data for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df["ctime"] = pd.to_datetime(df["time"], unit="s")
        df = df[["ctime", "open", "high", "low", "close", "tick_volume"]]
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# --- Neural Network Model (LSTM) ---
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def preprocess_data(df):
    if df.empty or len(df) < MIN_BARS:
        logger.error(f"DataFrame has {len(df)} bars, less than required {MIN_BARS}")
        return np.array([]), np.array([]), None
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["close"]].values)
    X, y = [], []
    for i in range(MIN_BARS, len(scaled_data)):
        X.append(scaled_data[i-MIN_BARS:i])
        if scaled_data[i] > scaled_data[i-1]:
            y.append([1, 0, 0])  # Up
        elif scaled_data[i] < scaled_data[i-1]:
            y.append([0, 1, 0])  # Down
        else:
            y.append([0, 0, 1])  # Sideways
    return np.array(X), np.array(y), scaler

# --- Risk Management ---
def calculate_atr(high, low, close, period=14):
    try:
        tr = np.maximum(high[1:] - low[1:],
                        np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:] - close[:-1]))
        atr = np.zeros(len(high))
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
        return atr
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return np.zeros(len(high))

def calculate_atr_stop_loss(df, period=14):
    if df.empty:
        logger.error("DataFrame is empty. Cannot calculate ATR")
        return 0
    atr = calculate_atr(df["high"].values, df["low"].values, df["close"].values, period)
    return atr[-1] * 2

def calculate_position_size(account_balance, symbol, sl_points):
    try:
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {symbol}")
            return symbol_info.volume_min
        
        tick_value = symbol_info.trade_tick_value
        risk_amount = account_balance * RISK_PER_TRADE
        volume = risk_amount / (sl_points * tick_value)
        volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
        volume = round(volume / symbol_info.volume_min) * symbol_info.volume_min
        return volume
    except Exception as e:
        logger.error(f"Error calculating position size for {symbol}: {e}")
        return symbol_info.volume_min

# --- Trading Logic ---
def predict_signal(model, X):
    try:
        if len(X) == 0:
            return [0.33, 0.33, 0.33]
        probs = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]), verbose=0)[0]
        return probs
    except Exception as e:
        logger.error(f"Error predicting signal: {e}")
        return [0.33, 0.33, 0.33]

def make_decision(signal_probs, position=None, current_price=None, tp=None, sl=None):
    up_prob, down_prob, side_prob = signal_probs
    direction = "Up" if up_prob == max(signal_probs) else "Down" if down_prob == max(signal_probs) else "Sideways"
    probs_str = f"[Up: {up_prob:.3f} | Down: {down_prob:.3f} | Side: {side_prob:.3f}]"

    if position is None:
        if up_prob > down_prob + PROB_THRESHOLD:
            decision = "BUY"
            reason = "Strong upward trend predicted"
        elif down_prob > up_prob + PROB_THRESHOLD:
            decision = "SELL"
            reason = "Strong downward trend predicted"
        else:
            decision = "HOLD"
            reason = "Trend unclear"
        confidence = max(signal_probs) * 100
        return decision, f"Predict: {probs_str} ({direction}) | Decision: {decision} | Reason: {reason}", confidence
    else:
        position_direction = "Up" if position.type == mt5.ORDER_TYPE_BUY else "Down"
        if (position.type == mt5.ORDER_TYPE_BUY and current_price >= tp) or \
           (position.type == mt5.ORDER_TYPE_SELL and current_price <= tp):
            decision = "TAKE PROFIT"
            reason = "Price reached TP"
        elif (position.type == mt5.ORDER_TYPE_BUY and current_price <= sl) or \
             (position.type == mt5.ORDER_TYPE_SELL and current_price >= sl):
            decision = "CUT LOSS"
            reason = "Price reached SL"
        elif direction == position_direction:
            decision = "HOLD"
            reason = "Trend aligns with position"
        else:
            decision = "CUT LOSS"
            reason = "Trend reversed"
        confidence = max(signal_probs) * 100
        return decision, f"Predict: {probs_str} ({direction}) | Decision: {decision} | Reason: {reason}", confidence

# --- Trade Execution ---
def send_trade(symbol, signal, tp, sl, volume):
    try:
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"Failed to get tick data for {symbol}")
            return None
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if signal == "BUY" else tick.bid,
            "sl": sl,
            "tp": tp,
            "comment": "AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Trade failed for {symbol}: {result.comment}, Retcode: {result.retcode}")
        else:
            logger.info(f"Trade executed: {signal} {volume} lots for {symbol}")
            notify(f"Trade executed: {signal} {volume} lots for {symbol}")
        return result
    except Exception as e:
        logger.error(f"Exception in send_trade for {symbol}: {e}")
        return None

def close_position(position):
    try:
        symbol = position.symbol
        ticket = position.ticket
        volume = position.volume
        position_type = position.type
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"Failed to get tick data for {symbol}")
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
            logger.error(f"Failed to close position {ticket}: {result.comment}")
        else:
            logger.info(f"Position {ticket} closed successfully")
            notify(f"Position {ticket} closed for {symbol}")
        return result
    except Exception as e:
        logger.error(f"Exception in close_position for {symbol}: {e}")
        return None

def update_trailing_stop(position, trailing_points=50):
    try:
        symbol = position.symbol
        symbol_info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if not tick or not symbol_info:
            logger.error(f"Failed to get tick or symbol info for {symbol}")
            return
        
        current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
        new_sl = current_price - trailing_points * symbol_info.point if position.type == mt5.ORDER_TYPE_BUY else \
                 current_price + trailing_points * symbol_info.point
        
        if (position.type == mt5.ORDER_TYPE_BUY and new_sl > position.sl) or \
           (position.type == mt5.ORDER_TYPE_SELL and new_sl < position.sl):
            mt5.position_modify(position.ticket, new_sl, position.tp)
            logger.info(f"Trailing stop updated for {symbol} to {new_sl}")
    except Exception as e:
        logger.error(f"Error updating trailing stop for {symbol}: {e}")

# --- Self-Learning Mechanism ---
def retrain_model(model, X, y):
    try:
        if len(X) == 0 or len(y) == 0:
            logger.error("No data to retrain model")
            return model
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                           epochs=10, batch_size=64, verbose=0)
        if history.history["val_loss"][-1] > 1.0:
            logger.warning("Retraining failed, reverting to previous model")
            if os.path.exists("best_model.h5"):
                return keras.models.load_model("best_model.h5")
        model.save("best_model.h5")
        logger.info("Model retrained and saved")
        return model
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        return model

# --- Performance Monitoring ---
def monitor_performance(account_balance, initial_balance):
    drawdown = (initial_balance - account_balance) / initial_balance
    if drawdown > MAX_DRAWDOWN:
        logger.error("Max drawdown exceeded. Stopping system")
        notify("EA stopped: Max drawdown exceeded")
        return False
    return True

# --- Backtesting Module ---
def backtest(symbol, start_date, end_date, initial_balance=10000):
    try:
        days = (end_date - start_date).days
        df = fetch_historical_data(symbol, days=days)
        if df.empty:
            logger.error(f"No data for backtesting {symbol}")
            return
        
        model = build_lstm_model((MIN_BARS, 1))
        X, y, scaler = preprocess_data(df)
        if len(X) > 0:
            model = retrain_model(model, X, y)
        
        equity = initial_balance
        trades = []
        
        for i in range(MIN_BARS, len(df)):
            X, _, _ = preprocess_data(df.iloc[:i])
            if len(X) == 0:
                continue
            signal_probs = predict_signal(model, X)
            decision, reason, confidence = make_decision(signal_probs)
            
            if decision in ["BUY", "SELL"]:
                atr_sl = calculate_atr_stop_loss(df.iloc[:i])
                current_price = df["close"].iloc[i]
                tp = current_price + atr_sl * 1.5 if decision == "BUY" else current_price - atr_sl * 1.5
                sl = current_price - atr_sl if decision == "BUY" else current_price + atr_sl
                volume = calculate_position_size(equity, symbol, atr_sl)
                
                # Simulate trade
                for j in range(i + 1, len(df)):
                    future_price = df["close"].iloc[j]
                    if decision == "BUY":
                        if future_price >= tp:
                            profit = (tp - current_price) * volume * 1000
                            equity += profit
                            trades.append({"entry": current_price, "exit": tp, "profit": profit})
                            break
                        elif future_price <= sl:
                            loss = (current_price - sl) * volume * 1000
                            equity -= loss
                            trades.append({"entry": current_price, "exit": sl, "profit": -loss})
                            break
                    else:
                        if future_price <= tp:
                            profit = (current_price - tp) * volume * 1000
                            equity += profit
                            trades.append({"entry": current_price, "exit": tp, "profit": profit})
                            break
                        elif future_price >= sl:
                            loss = (sl - current_price) * volume * 1000
                            equity -= loss
                            trades.append({"entry": current_price, "exit": sl, "profit": -loss})
                            break
        
        logger.info(f"Backtest result for {symbol}: Final equity = {equity}, Trades = {len(trades)}")
        return equity, trades
    except Exception as e:
        logger.error(f"Error in backtesting for {symbol}: {e}")
        return initial_balance, []

# --- Main Trading Loop ---
def main():
    if not init_mt5():
        notify("EA failed to initialize")
        return
    
    # Test Telegram connection
    notify("EA started successfully")
    
    # Run backtest before live trading
    for symbol in SYMBOLS:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        equity, trades = backtest(symbol, start_date, end_date)
        logger.info(f"Backtest for {symbol}: Equity = {equity}, Trades = {len(trades)}")
    
    account_info = mt5.account_info()
    initial_balance = account_info.balance
    last_retrain = datetime.now() - timedelta(days=RETRAIN_INTERVAL + 1)
    model = build_lstm_model((MIN_BARS, 1))
    
    while True:
        try:
            current_time = datetime.now()
            
            # Retrain model periodically
            if (current_time - last_retrain).days >= RETRAIN_INTERVAL:
                logger.info("Retraining model...")
                for symbol in SYMBOLS:
                    df = fetch_historical_data(symbol)
                    X, y, scaler = preprocess_data(df)
                    if len(X) > 0:
                        model = retrain_model(model, X, y)
                last_retrain = current_time
            
            # Fetch news
            high_impact_news = fetch_high_impact_news()
            trading_allowed = True
            
            for event in high_impact_news:
                time_to_news = (event["time"] - current_time).total_seconds()
                if 0 < time_to_news <= NEWS_STOP_MINUTES * 60:
                    trading_allowed = False
                    logger.info(f"News in {time_to_news:.0f} seconds: {event['name']}. Pausing trades")
                    break
            
            for symbol in SYMBOLS:
                df = fetch_historical_data(symbol)
                X, _, scaler = preprocess_data(df)
                if len(X) == 0:
                    logger.warning(f"No valid data for {symbol}. Skipping")
                    continue
                
                signal_probs = predict_signal(model, X)
                position = mt5.positions_get(symbol=symbol)
                position = position[0] if position else None
                
                atr_sl = calculate_atr_stop_loss(df)
                current_price = df["close"].iloc[-1] if not position else \
                               mt5.symbol_info_tick(symbol).bid if position.type == mt5.ORDER_TYPE_BUY else \
                               mt5.symbol_info_tick(symbol).ask
                
                if position:
                    tp, sl = position.tp, position.sl
                    decision, reasoning, confidence = make_decision(signal_probs, position, current_price, tp, sl)
                    
                    position_signal = {
                        "pair": symbol,
                        "current_price": float(current_price),
                        "sl": float(sl),
                        "tp": float(tp),
                        "decision": decision,
                        "confidence": float(confidence),
                        "reason": reasoning
                    }
                    logger.info(json.dumps(position_signal, indent=2))
                    
                    if decision in ["CUT LOSS", "TAKE PROFIT"]:
                        close_position(position)
                    else:
                        update_trailing_stop(position)
                else:
                    if not trading_allowed:
                        logger.info(f"Trading paused for {symbol} due to news")
                        continue
                    
                    decision, reasoning, confidence = make_decision(signal_probs)
                    tp = current_price + atr_sl * 1.5 if decision == "BUY" else current_price - atr_sl * 1.5
                    sl = current_price - atr_sl if decision == "BUY" else current_price + atr_sl
                    account_info = mt5.account_info()
                    volume = calculate_position_size(account_info.balance, symbol, atr_sl)
                    
                    trade_signal = {
                        "pair": symbol,
                        "signal": decision,
                        "confidence": float(confidence),
                        "reason": reasoning,
                        "tp": float(tp),
                        "sl": float(sl),
                        "volume": float(volume)
                    }
                    logger.info(json.dumps(trade_signal, indent=2))
                    
                    if decision in ["BUY", "SELL"]:
                        send_trade(symbol, decision, tp, sl, volume)
            
            if not monitor_performance(account_info.balance, initial_balance):
                notify("EA stopped due to max drawdown")
                return
            
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            notify(f"EA error: {str(e)}")
            time.sleep(300)  # Wait 5 minutes before retry

if __name__ == "__main__":
    main()
