#!/usr/bin/env python3
"""
ultimate_stock_toolkit_v2.py

Unified advanced stock toolkit:
- Data fetch (yfinance)
- Anomaly detection
- Sentiment (NewsAPI/TextBlob)
- Portfolio optimization
- Monte Carlo sims
- Black-Scholes options
- GARCH volatility
- RL trading signal stub
- SMA strategy + backtesting + performance metrics
- Telegram + Email + Webhook alerts
- SQLite logging (SQLAlchemy)
- Dash dashboard (optional)
- Scheduler + CLI

Install:
pip install yfinance pandas numpy textblob telegram dash plotly sqlalchemy requests pyyaml arch scipy
"""

import os
import sys
import time
import math
import json
import yaml
import sqlite3
import threading
import argparse
import requests
import smtplib
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import List, Dict, Tuple

import yfinance as yf
import numpy as np
import pandas as pd
from textblob import TextBlob
import telegram
from arch import arch_model
from scipy.stats import norm
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Table, MetaData

# Optional dash imports (only used when --dashboard passed)
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.graph_objs as go
    DASH_AVAILABLE = True
except Exception:
    DASH_AVAILABLE = False

warnings.filterwarnings("ignore")

# --------------------------
# CONFIG (YAML file support)
# --------------------------
DEFAULT_CONFIG = {
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "lookback_days": 90,
    "anomaly_threshold_std": 2.0,
    "telegram": {"token": "", "chat_id": ""},
    "smtp": {"host": "", "port": 587, "user": "", "pass": "", "from": "", "to": ""},
    "newsapi_key": "",
    "db": {"sqlite_path": "stock_toolkit.db"},
    "webhook_url": "",
    "transaction_cost": 0.001,  # 0.1%
    "backtest": {"initial_capital": 100000},
    "dash": {"host": "127.0.0.1", "port": 8050}
}

def load_config(path: str = "config.yml") -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
            merged = DEFAULT_CONFIG.copy()
            merged.update(cfg or {})
            return merged
    return DEFAULT_CONFIG.copy()

CFG = load_config()

# --------------------------
# DATABASE (SQLAlchemy Lightweight)
# --------------------------
engine = create_engine(f"sqlite:///{CFG['db']['sqlite_path']}", echo=False)
meta = MetaData()

alerts_table = Table(
    "alerts", meta,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("ticker", String),
    Column("type", String),
    Column("message", Text),
    Column("payload", Text)
)
backtests_table = Table(
    "backtests", meta,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("ticker", String),
    Column("strategy", String),
    Column("initial_capital", Float),
    Column("final_capital", Float),
    Column("metrics_json", Text),
    Column("details", Text)
)

meta.create_all(engine)

def db_insert_alert(ticker, type_, message, payload):
    ins = alerts_table.insert().values(created_at=datetime.utcnow(), ticker=ticker, type=type_, message=message, payload=json.dumps(payload))
    conn = engine.connect()
    conn.execute(ins)
    conn.close()

def db_insert_backtest(record: dict):
    ins = backtests_table.insert().values(
        created_at=datetime.utcnow(), ticker=record["ticker"], strategy=record["strategy"],
        initial_capital=record["initial_capital"], final_capital=record["final_capital"],
        metrics_json=json.dumps(record["metrics"]), details=json.dumps(record.get("details", {}))
    )
    conn = engine.connect()
    conn.execute(ins)
    conn.close()

# --------------------------
# UTIL: fetch data
# --------------------------
def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df = df.dropna()
    df["Returns"] = df["Close"].pct_change()
    return df

# --------------------------
# ANOMALY DETECTION
# --------------------------
def detect_anomalies(df: pd.DataFrame, column: str = "Returns", threshold: float = None) -> pd.DataFrame:
    if threshold is None:
        threshold = CFG["anomaly_threshold_std"]
    mean = df[column].mean()
    std = df[column].std()
    df["Anomaly"] = (df[column] - mean).abs() > threshold * std
    return df

# --------------------------
# NEWS SENTIMENT
# --------------------------
def fetch_news_headlines(query: str, api_key: str = None, limit: int = 5) -> List[Dict]:
    if not api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": limit, "apiKey": api_key}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return []
    items = r.json().get("articles", [])
    return [{"title": it["title"], "publishedAt": it["publishedAt"], "url": it["url"]} for it in items]

def analyze_sentiment_text(text: str) -> float:
    return TextBlob(text).sentiment.polarity

def stock_news_sentiment(ticker: str) -> Dict:
    articles = fetch_news_headlines(ticker, CFG.get("newsapi_key"), limit=5)
    sentiments = []
    for a in articles:
        p = analyze_sentiment_text(a["title"])
        sentiments.append({"title": a["title"], "polarity": p, "publishedAt": a["publishedAt"], "url": a["url"]})
    score = np.mean([s["polarity"] for s in sentiments]) if sentiments else 0.0
    return {"score": float(score), "articles": sentiments}

# --------------------------
# GARCH Volatility Forecast
# --------------------------
def garch_vol_forecast(ticker: str, horizon: int = 5) -> pd.Series:
    df = fetch_ohlcv(ticker, period="2y", interval="1d")
    returns = 100 * df["Close"].pct_change().dropna()
    model = arch_model(returns, vol="Garch", p=1, q=1, mean="Zero")
    res = model.fit(disp="off")
    fc = res.forecast(horizon=horizon)
    # return sqrt of variance (percent)
    var = fc.variance.iloc[-1]
    vol = np.sqrt(var)
    return vol

# --------------------------
# Black-Scholes Options
# --------------------------
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        # already expired
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(price)

# --------------------------
# Monte Carlo Simulation
# --------------------------
def monte_carlo_paths(ticker: str, days: int = 30, sims: int = 500) -> np.ndarray:
    df = fetch_ohlcv(ticker, period="1y", interval="1d")
    returns = df["Close"].pct_change().dropna()
    S0 = float(df["Close"].iloc[-1])
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    paths = np.zeros((sims, days + 1))
    for i in range(sims):
        prices = [S0]
        for _ in range(days):
            shock = np.random.normal((mu - 0.5 * sigma ** 2) / 252, sigma / np.sqrt(252))
            prices.append(prices[-1] * math.exp(shock))
        paths[i, :] = prices
    return paths

# --------------------------
# Portfolio optimization (Monte-Carlo naive)
# --------------------------
def optimize_portfolio_montecarlo(tickers: List[str], n_samples: int = 5000):
    df = yf.download(tickers, period="1y")["Adj Close"].dropna()
    returns = df.pct_change().dropna()
    mu = returns.mean()
    cov = returns.cov()
    num = n_samples
    results = np.zeros((num, 3))
    all_weights = []
    for i in range(num):
        w = np.random.random(len(tickers))
        w /= w.sum()
        all_weights.append(w)
        port_ret = np.sum(w * mu) * 252
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov * 252, w)))
        results[i, 0] = port_ret
        results[i, 1] = port_vol
        results[i, 2] = results[i, 0] / results[i, 1]  # sharpe
    idx = results[:, 2].argmax()
    return {"weights": list(map(float, all_weights[idx])), "return": float(results[idx, 0]), "volatility": float(results[idx, 1]), "sharpe": float(results[idx, 2])}

# --------------------------
# RL Trading Signal (lightweight stub)
# --------------------------
def rl_signal_stub(ticker: str) -> int:
    # returns: 0 hold, 1 buy, 2 sell
    # This is a placeholder to plug a trained RL agent
    df = fetch_ohlcv(ticker, period="1y")
    recent = df["Close"].pct_change().tail(5).mean()
    if recent > 0.01:
        return 1
    elif recent < -0.01:
        return 2
    return 0

# --------------------------
# Strategy: SMA crossover (simple)
# --------------------------
@dataclass
class Trade:
    date: datetime
    action: str
    price: float
    size: float
    cash: float
    position: float

def sma_crossover_signals(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    df = df.copy()
    df["SMA_short"] = df["Close"].rolling(window=short).mean()
    df["SMA_long"] = df["Close"].rolling(window=long).mean()
    df["signal"] = 0
    df.loc[df["SMA_short"] > df["SMA_long"], "signal"] = 1
    df.loc[df["SMA_short"] < df["SMA_long"], "signal"] = -1
    df["signal_change"] = df["signal"].diff().fillna(0)
    return df

# --------------------------
# Backtester
# --------------------------
def backtest_strategy(df: pd.DataFrame, strategy_fn, initial_capital: float = 100000.0, transaction_cost: float = 0.001) -> Tuple[dict, List[Trade]]:
    """
    Generic backtester expects df with datetime index and 'Close' column.
    strategy_fn should annotate df with 'signal_change' where:
    +2 or +1 => buy signal, -2 or -1 => sell signal
    """
    df = df.copy().dropna()
    signals = strategy_fn(df)
    cash = initial_capital
    position = 0.0  # number of shares
    trades: List[Trade] = []

    for idx, row in signals.iterrows():
        s = int(row.get("signal_change", 0))
        price = float(row["Close"])
        # Buy on rising crossover
        if s > 0 and cash > price:
            # all-in simplistic sizing (can be improved)
            size = (cash / price)  # buy as many as possible
            cost = size * price * (1 + transaction_cost)
            position += size
            cash -= cost
            trades.append(Trade(date=idx.to_pydatetime(), action="BUY", price=price, size=size, cash=cash, position=position))
        elif s < 0 and position > 0:
            # sell all
            proceeds = position * price * (1 - transaction_cost)
            cash += proceeds
            trades.append(Trade(date=idx.to_pydatetime(), action="SELL", price=price, size=position, cash=cash, position=0.0))
            position = 0.0
    # Liquidate at end
    if position > 0:
        final_price = float(signals["Close"].iloc[-1])
        proceeds = position * final_price * (1 - transaction_cost)
        cash += proceeds
        trades.append(Trade(date=signals.index[-1].to_pydatetime(), action="SELL_AT_END", price=final_price, size=position, cash=cash, position=0.0))
        position = 0.0

    final_capital = cash
    metrics = compute_performance_metrics_from_trades(trades, initial_capital)
    return {"initial_capital": initial_capital, "final_capital": final_capital, "metrics": metrics}, trades

# --------------------------
# Performance Metrics utilities
# --------------------------
def compute_performance_metrics_from_trades(trades: List[Trade], initial_capital: float) -> dict:
    if not trades:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDrawdown": 0.0, "Trades": 0, "WinRate": 0.0}
    # Build equity curve from trades
    # Simplistic: compute returns at trade points only
    equity = []
    capital = initial_capital
    position = 0
    last_date = trades[0].date
    # We will simulate by applying trades sequentially
    vals = [initial_capital]
    for t in trades:
        vals.append(t.cash)
    arr = np.array(vals)
    returns = pd.Series(arr).pct_change().fillna(0)
    total_return = (arr[-1] / initial_capital) - 1
    years = max(1/252, (trades[-1].date - trades[0].date).days / 365)
    cagr = (arr[-1] / initial_capital) ** (1 / years) - 1
    sharpe = returns.mean() / (returns.std() + 1e-9) * math.sqrt(252) if returns.std() > 0 else 0.0
    # Max drawdown (simple)
    peak = arr[0]
    maxdd = 0.0
    for v in arr:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > maxdd:
            maxdd = dd
    # Win rate
    wins = 0
    total_pairs = 0
    buy_price = None
    for t in trades:
        if t.action == "BUY":
            buy_price = t.price
        elif t.action.startswith("SELL") and buy_price is not None:
            total_pairs += 1
            if t.price > buy_price:
                wins += 1
            buy_price = None
    winrate = wins / total_pairs if total_pairs > 0 else 0.0
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDrawdown": float(maxdd), "Trades": len(trades), "WinRate": float(winrate), "TotalReturn": float(total_return)}

# --------------------------
# Alerts: Telegram, Email, Webhook
# --------------------------
TELEGRAM_ENABLED = bool(CFG.get("telegram", {}).get("token")) and bool(CFG.get("telegram", {}).get("chat_id"))
if TELEGRAM_ENABLED:
    tg_bot = telegram.Bot(token=CFG["telegram"]["token"])
else:
    tg_bot = None

def send_telegram(text: str):
    if not tg_bot:
        return
    try:
        tg_bot.send_message(chat_id=CFG["telegram"]["chat_id"], text=text)
    except Exception as e:
        print("Telegram send error:", e)

def send_email(subject: str, body: str):
    smtp = CFG.get("smtp", {})
    if not smtp.get("host"):
        return
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = smtp.get("from", smtp.get("user"))
    msg["To"] = smtp.get("to")
    try:
        s = smtplib.SMTP(smtp["host"], smtp.get("port", 587), timeout=10)
        s.starttls()
        s.login(smtp["user"], smtp["pass"])
        s.sendmail(msg["From"], [msg["To"]], msg.as_string())
        s.quit()
    except Exception as e:
        print("Email error:", e)

def send_webhook(payload: dict):
    url = CFG.get("webhook_url")
    if not url:
        return
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print("Webhook error:", e)

def emit_alert(ticker: str, alert_type: str, message: str, payload: dict = None):
    print(f"[ALERT] {ticker} | {alert_type} | {message}")
    db_insert_alert(ticker, alert_type, message, payload or {})
    send_telegram(f"{ticker} | {alert_type} | {message}")
    send_email(f"{ticker} - {alert_type}", message)
    send_webhook({"ticker": ticker, "type": alert_type, "message": message, "payload": payload or {}})

# --------------------------
# Scheduler + Monitor
# --------------------------
def monitor_and_alert_once(tickers: List[str]):
    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, period=f"{CFG['lookback_days']}d", interval="1d")
            df = detect_anomalies(df, CFG["anomaly_threshold_std"])
            if df["Anomaly"].iloc[-1]:
                # anomaly detected
                sentiment = stock_news_sentiment(ticker)
                emit_alert(ticker, "ANOMALY", f"Anomaly detected for {ticker} on {df.index[-1].date()}; sentiment:{sentiment['score']:.2f}", {"sentiment": sentiment})
            # check SMA signal
            s = sma_crossover_signals(df, short=20, long=50)
            if s["signal_change"].iloc[-1] > 0:
                emit_alert(ticker, "SMA_SIGNAL", "Bullish SMA crossover detected", {"last_close": float(df['Close'].iloc[-1])})
            elif s["signal_change"].iloc[-1] < 0:
                emit_alert(ticker, "SMA_SIGNAL", "Bearish SMA crossover detected", {"last_close": float(df['Close'].iloc[-1])})
        except Exception as e:
            print("Monitor error for", ticker, e)
            traceback.print_exc()

def scheduled_monitor_loop(interval_seconds: int = 3600):
    print("Starting scheduled monitoring loop...")
    while True:
        monitor_and_alert_once(CFG["tickers"])
        time.sleep(interval_seconds)

# --------------------------
# CLI / Modes
# --------------------------
def run_backtest_mode(tickers: List[str]):
    for ticker in tickers:
        df = fetch_ohlcv(ticker, period="2y", interval="1d")
        result, trades = backtest_strategy(df, lambda d: sma_crossover_signals(d, short=20, long=50), initial_capital=CFG["backtest"]["initial_capital"], transaction_cost=CFG["transaction_cost"])
        db_insert_backtest({"ticker": ticker, "strategy": "SMA_20_50", "initial_capital": result["initial_capital"], "final_capital": result["final_capital"], "metrics": result["metrics"], "details": {"trades": [t.__dict__ for t in trades]}})
        print(f"Backtest {ticker}: initial {result['initial_capital']:.2f} -> final {result['final_capital']:.2f}")
        print("Metrics:", result["metrics"])

def run_scan_mode(tickers: List[str]):
    monitor_and_alert_once(tickers)

# --------------------------
# DASHBOARD (optional)
# --------------------------
def run_dash_app(tickers: List[str]):
    if not DASH_AVAILABLE:
        print("Dash not installed. Install dash and plotly to use dashboard.")
        return
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Ultimate Stock Toolkit V2"),
        dcc.Dropdown(id="ticker_select", options=[{"label": t, "value": t} for t in tickers], value=tickers[0]),
        dcc.Graph(id="price_chart"),
        dcc.Interval(id="interval", interval=60*1000, n_intervals=0)
    ])

    @app.callback(Output("price_chart", "figure"), [Input("ticker_select", "value"), Input("interval", "n_intervals")])
    def update_chart(ticker, n):
        df = fetch_ohlcv(ticker, period="6mo", interval="1d")
        df = detect_anomalies(df)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
        anomalies = df[df["Anomaly"]]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies["Close"], mode="markers", name="Anomaly", marker=dict(color="red", size=8)))
        # SMA
        df["sma20"] = df["Close"].rolling(20).mean()
        df["sma50"] = df["Close"].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=df["sma20"], mode="lines", name="SMA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["sma50"], mode="lines", name="SMA50"))
        return fig

    app.run_server(host=CFG["dash"]["host"], port=CFG["dash"]["port"], debug=False)

# --------------------------
# MAIN
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Ultimate Stock Toolkit V2")
    parser.add_argument("--mode", choices=["scan", "backtest", "dashboard", "monitor"], default="scan")
    parser.add_argument("--tickers", nargs="+", default=CFG["tickers"])
    parser.add_argument("--once", action="store_true", help="Run once and exit (for monitor mode)")
    parser.add_argument("--interval", type=int, default=3600, help="Monitor interval seconds")
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest_mode(args.tickers)
    elif args.mode == "dashboard":
        run_dash_app(args.tickers)
    elif args.mode == "monitor":
        if args.once:
            monitor_and_alert_once(args.tickers)
        else:
            scheduled_monitor_loop(interval_seconds=args.interval)
    else:  # scan
        run_scan_mode(args.tickers)

if __name__ == "__main__":
    main()
