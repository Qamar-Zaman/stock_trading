import logging
import sqlite3
import time

import pandas as pd
import talib
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Database setup
def initialize_db(db_name='trades.db'):
    """Initialize SQLite database and create trades table."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            profit REAL,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized.")


# Record a trade
def record_trade(conn, timestamp, symbol, direction, entry_price, exit_price, profit, status):
    """Insert a trade into the database with an active connection."""
    try:
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, pd.Timestamp) else timestamp
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, direction, entry_price, exit_price, profit, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, symbol, direction, entry_price, exit_price or 'None', profit or 'None', status))
        conn.commit()
        logger.info(
            f"TRADE EXECUTED | {direction} | Entry: {entry_price} | Exit: {exit_price} | Profit: {profit} | Status: {status}")
    except Exception as e:
        logger.error(f"Error recording trade: {e}")


# Download market data
def download_data(symbol, start_date, end_date, interval='1d'):
    try:
        logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}...")
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)

        if data.empty:
            logger.error(f"No data found for {symbol}. Check the ticker and date range.")
            return None

        logger.info(f"Data downloaded successfully. Rows: {len(data)}")

        # Flatten multi-index column names
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        if not {'Open', 'High', 'Low', 'Close', 'Volume'}.issubset(data.columns):
            logger.error(f"Missing necessary columns in downloaded data. Columns found: {data.columns.tolist()}")
            return None

        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return None


# Calculate Alligator Indicator
def calculate_alligator(data):
    data['Jaw'] = data['Close'].ewm(span=13, adjust=False).mean().shift(8)
    data['Teeth'] = data['Close'].ewm(span=8, adjust=False).mean().shift(5)
    data['Lips'] = data['Close'].ewm(span=5, adjust=False).mean().shift(3)
    return data


# Calculate additional indicators
def calculate_indicators(data):
    high, low, close, volume = data['High'], data['Low'], data['Close'], data['Volume']
    data['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    data['CMO'] = talib.CMO(close, timeperiod=14)
    data['Stoch_RSI'], _ = talib.STOCHRSI(close, timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)
    data['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    data['Volume_Change'] = volume.pct_change()
    data['EMA_50'] = talib.EMA(close, timeperiod=50)
    data['MACD'], data['MACD_signal'], _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    data = data.ffill()  # Fill NaN values
    return data


# Generate trade signals with trend confirmation
def generate_signals(data):
    data['Signal'] = 0
    long_condition = (
            (data['Lips'] > data['Teeth']) & (data['Teeth'] > data['Jaw']) & (data['Close'] > data['Lips']) & (
                data['ADX'] > 30) & (data['CMO'] > 40) & (data['Stoch_RSI'] > 60) & (data['Volume_Change'] > 0) & (
                        data['Close'] > data['EMA_50']) & (data['MACD'] > data['MACD_signal']))
    short_condition = (
            (data['Lips'] < data['Teeth']) & (data['Teeth'] < data['Jaw']) & (data['Close'] < data['Lips']) & (
                data['ADX'] > 30) & (data['CMO'] < -40) & (data['Stoch_RSI'] < 40) & (data['Volume_Change'] < 0) & (
                        data['Close'] < data['EMA_50']) & (data['MACD'] < data['MACD_signal']))

    data.loc[long_condition, 'Signal'] = 1
    data.loc[short_condition, 'Signal'] = -1
    return data


# Backtesting with adaptive stop-loss and trailing stop
def backtest_strategy(data, symbol, initial_capital=1000, risk_per_trade=0.03):
    conn = sqlite3.connect('trades.db')
    position, portfolio = 0, initial_capital
    trades, entry_price, trailing_stop = [], 0, 0

    for i in range(len(data)):
        atr = data['ATR'].iat[i]
        volatility_factor = atr / data['Close'].iat[i]
        if volatility_factor < 0.01:
            stop_loss = atr * 2
            take_profit = atr * 4
        elif volatility_factor > 0.02:
            stop_loss = atr * 1.5
            take_profit = atr * 3
        else:
            stop_loss = atr * 1.75
            take_profit = atr * 3.5

        # Entry Conditions
        if data['Signal'].iat[i] == 1 and position == 0:  # Long Entry
            position_size_factor = max(0.5, min(1.5, 0.02 / volatility_factor))
            position = (initial_capital * risk_per_trade * position_size_factor) / stop_loss
            entry_price = data['Close'].iat[i]
            trailing_stop = entry_price - atr
            record_trade(conn, data.index[i], symbol, 'Long', entry_price, None, None, 'Open')
            trades.append(
                {'timestamp': data.index[i], 'symbol': symbol, 'direction': 'Long', 'entry_price': entry_price,
                 'exit_price': None, 'profit': None, 'status': 'Open'})

        elif data['Signal'].iat[i] == -1 and position == 0:  # Short Entry
            position_size_factor = max(0.5, min(1.5, 0.02 / volatility_factor))
            position = - (initial_capital * risk_per_trade * position_size_factor) / stop_loss
            entry_price = data['Close'].iat[i]
            trailing_stop = entry_price + atr
            record_trade(conn, data.index[i], symbol, 'Short', entry_price, None, None, 'Open')
            trades.append(
                {'timestamp': data.index[i], 'symbol': symbol, 'direction': 'Short', 'entry_price': entry_price,
                 'exit_price': None, 'profit': None, 'status': 'Open'})

        # Exit Conditions
        elif position != 0:
            exit_price = data['Close'].iat[i]
            # Update trailing stop based on ATR and price movement
            if position > 0 and exit_price > entry_price + atr:
                trailing_stop = max(trailing_stop, exit_price - atr * 1.5)
            elif position < 0 and exit_price < entry_price - atr:
                trailing_stop = min(trailing_stop, exit_price + atr * 1.5)

            if position > 0 and (exit_price >= entry_price + take_profit or exit_price <= trailing_stop):  # Close Long
                profit = (exit_price - entry_price) * position
                record_trade(conn, data.index[i], symbol, 'Long', entry_price, exit_price, profit, 'Closed')
                portfolio += profit
                position = 0
                trades[-1].update({'exit_price': exit_price, 'profit': profit, 'status': 'Closed'})
            elif position < 0 and (
                    exit_price <= entry_price - take_profit or exit_price >= trailing_stop):  # Close Short
                profit = (entry_price - exit_price) * abs(position)
                record_trade(conn, data.index[i], symbol, 'Short', entry_price, exit_price, profit, 'Closed')
                portfolio += profit
                position = 0
                trades[-1].update({'exit_price': exit_price, 'profit': profit, 'status': 'Closed'})

    # Forced Closure at End of Backtest
    if position != 0:
        logger.warning(f"Trade still open at the end of backtest: {trades[-1]}")
        exit_price = data['Close'].iat[-1]  # Use last close price
        profit = (exit_price - entry_price) * position if position > 0 else (entry_price - exit_price) * abs(position)
        portfolio += profit
        trades[-1].update({'exit_price': exit_price, 'profit': profit, 'status': 'Closed'})
        logger.warning(f"Forced exit at {exit_price} (last price). Profit: {profit}. Portfolio: {portfolio}")
        record_trade(conn, data.index[-1], symbol, trades[-1]['direction'], entry_price, exit_price, profit, 'Closed')

    conn.close()
    return trades, portfolio


# Main function
def main():
    initial_capital = 10000
    initialize_db()
    # symbol, start_date, end_date = 'BTC-USD', '2024-01-01', '2025-03-01'
    # symbol, start_date, end_date = 'ETH-USD', '2024-01-01', '2025-03-01'
    # symbol, start_date, end_date = 'XRP-USD', '2024-01-01', '2025-03-01'
    # symbol, start_date, end_date = 'AAPL', '2024-01-01', '2025-03-01'
    # symbol, start_date, end_date = 'MSTR', '2024-01-01', '2025-03-01'
    # symbol, start_date, end_date = 'NVDA', '2024-01-01', '2025-03-01'

    symbols = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'AAPL', 'MSTR', 'NVDA']
    start_date, end_date = '2024-01-01', '2025-03-21'

    for symbol in symbols:
        data = download_data(symbol, start_date, end_date)
        if data is not None:
            data = calculate_alligator(data)
            data = calculate_indicators(data)
            data = generate_signals(data)
            trades, portfolio = backtest_strategy(data, symbol, initial_capital=initial_capital)

            logger.info(
                f"FINAL RESULTS: Symbol: {symbol} | Total trades: {len(trades)} | Start Date: {start_date} | End Date: {end_date} | Portfolio Value: {portfolio:.2f} | Initial Capital: {initial_capital:.2f} | Net P&L: {portfolio - initial_capital:.2f}")
        else:
            logger.error(f"No Data available for {symbol}, Start Date: {start_date}, End Date: {end_date}")
        time.sleep(2)


if __name__ == "__main__":
    main()
