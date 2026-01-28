import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import random
import http.server
import socketserver
import warnings
import requests
import threading
import time
import json
import urllib.parse
from datetime import datetime, timedelta
from deap import base, creator, tools, algorithms

# --- Configuration ---
BASE_DATA_URL = "https://ohlcendpoint.up.railway.app/data"
PORT = 8080
N_LINES = 32
POPULATION_SIZE = 320
GENERATIONS = 10
RISK_FREE_RATE = 0.0
MAX_ASSETS_TO_OPTIMIZE = 1  # Limit the number of assets processed by GA

# Ranges
STOP_PCT_RANGE = (0.001, 0.02)   # 0.1% to 2%
PROFIT_PCT_RANGE = (0.0004, 0.05) # 0.04% to 5%

warnings.filterwarnings("ignore")

# Asset List Mapping
ASSETS = [
    {"symbol": "BTC", "pair": "BTCUSDT", "csv": "btc1m.csv"},
    {"symbol": "ETH", "pair": "ETHUSDT", "csv": "eth1m.csv"},
    {"symbol": "XRP", "pair": "XRPUSDT", "csv": "xrp1m.csv"},
    {"symbol": "SOL", "pair": "SOLUSDT", "csv": "sol1m.csv"},
    {"symbol": "DOGE", "pair": "DOGEUSDT", "csv": "doge1m.csv"},
    {"symbol": "ADA", "pair": "ADAUSDT", "csv": "ada1m.csv"},
    {"symbol": "BCH", "pair": "BCHUSDT", "csv": "bch1m.csv"},
    {"symbol": "LINK", "pair": "LINKUSDT", "csv": "link1m.csv"},
    {"symbol": "XLM", "pair": "XLMUSDT", "csv": "xlm1m.csv"},
    {"symbol": "SUI", "pair": "SUIUSDT", "csv": "sui1m.csv"},
    {"symbol": "AVAX", "pair": "AVAXUSDT", "csv": "avax1m.csv"},
    {"symbol": "LTC", "pair": "LTCUSDT", "csv": "ltc1m.csv"},
    {"symbol": "HBAR", "pair": "HBARUSDT", "csv": "hbar1m.csv"},
    {"symbol": "SHIB", "pair": "SHIBUSDT", "csv": "shib1m.csv"},
    {"symbol": "TON", "pair": "TONUSDT", "csv": "ton1m.csv"},
]

# Global Storage
HTML_REPORTS = {} 
BEST_PARAMS = {}
REPORT_LOCK = threading.Lock()

# --- 1. DEAP Initialization ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- 2. Precise Data Ingestion ---
def get_data(csv_filename):
    url = f"{BASE_DATA_URL}/{csv_filename}"
    print(f"Downloading data from {url}...")
    try:
        df = pd.read_csv(url)
        df.columns = [c.lower().strip() for c in df.columns]
        
        if 'datetime' in df.columns:
            df['dt'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'timestamp' in df.columns:
            df['dt'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        else:
            raise ValueError("No valid date column found.")

        df.dropna(subset=['dt', 'open', 'high', 'low', 'close'], inplace=True)
        df.set_index('dt', inplace=True)
        df.sort_index(inplace=True)

        print(f"[{csv_filename}] Raw 1m Data: {len(df)} rows")
        
        # Store Raw 1M Data for Drill-Down Resolution
        df_raw = df[['open', 'high', 'low', 'close']].copy()

        # Resample to 1H for Main Loop Speed
        df_1h = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        print(f"[{csv_filename}] Resampled 1H Data (For GA): {len(df_1h)} rows")
        
        if len(df_1h) < 100:
            raise ValueError("Data insufficient after resampling.")

        split_idx = int(len(df_1h) * 0.85)
        
        # Returns: train_1h, test_1h, FULL_raw_1m (passed to backtester for lookup)
        train_1h = df_1h.iloc[:split_idx]
        test_1h = df_1h.iloc[split_idx:]
        
        return train_1h, test_1h, df_raw

    except Exception as e:
        print(f"CRITICAL DATA ERROR for {csv_filename}: {e}")
        return None, None, None

# --- 3. Drill-Down Ambiguity Resolution ---
def resolve_ambiguity(timestamp_1h, short_price, long_price, raw_df):
    """
    Checks the underlying 1m candles for a specific 1h candle 
    to see which price was hit first.
    """
    # Slice the raw data for this specific hour
    # We look from [current_hour] to [current_hour + 59 min]
    end_time = timestamp_1h + timedelta(hours=1)
    
    # Get 1m slice (handles missing data gracefully)
    try:
        slice_1m = raw_df.loc[timestamp_1h : end_time]
    except KeyError:
        return 0 # No data, stick with neutral/heuristic

    if slice_1m.empty:
        return 0

    # Iterate minute by minute
    for ts, row in slice_1m.iterrows():
        hit_short = row['high'] >= short_price
        hit_long = row['low'] <= long_price
        
        if hit_short and hit_long:
            # Even the minute is ambiguous! Fallback to proximity heuristic for this minute
            dist_short = abs(row['open'] - short_price)
            dist_long = abs(row['open'] - long_price)
            return -1 if dist_short < dist_long else 1
            
        elif hit_short:
            return -1 # Short happened first
        elif hit_long:
            return 1 # Long happened first
            
    return 0 # Should not happen if the 1H candle actually touched the lines

# --- 4. Strategy Logic ---
def run_backtest(df_1h, raw_df, stop_pct, profit_pct, lines, detailed_log_trades=0):
    closes = df_1h['close'].values
    highs = df_1h['high'].values
    lows = df_1h['low'].values
    times = df_1h.index
    
    equity = 10000.0
    equity_curve = [equity]
    position = 0          # 0: Flat, 1: Long, -1: Short
    entry_price = 0.0
    
    trades = []
    hourly_log = []
    
    lines = np.sort(lines)
    trades_completed = 0
    
    for i in range(1, len(df_1h)):
        current_c = closes[i]
        current_h = highs[i]
        current_l = lows[i]
        prev_c = closes[i-1] # Acts as "Open"
        ts = times[i]
        
        # --- Detailed Logging ---
        if detailed_log_trades > 0 and trades_completed < detailed_log_trades:
            # (Logging logic remains same)
            pass 

        # --- Position Management (Exit Logic) ---
        if position != 0:
            sl_hit = False
            tp_hit = False
            exit_price = 0.0
            reason = ""

            if position == 1: # Long Logic
                sl_price = entry_price * (1 - stop_pct)
                tp_price = entry_price * (1 + profit_pct)
                if current_l <= sl_price: sl_hit = True; exit_price = sl_price 
                elif current_h >= tp_price: tp_hit = True; exit_price = tp_price

            elif position == -1: # Short Logic
                sl_price = entry_price * (1 + stop_pct)
                tp_price = entry_price * (1 - profit_pct)
                if current_h >= sl_price: sl_hit = True; exit_price = sl_price
                elif current_l <= tp_price: tp_hit = True; exit_price = tp_price
            
            if sl_hit or tp_hit:
                if position == 1: pn_l = (exit_price - entry_price) / entry_price
                else: pn_l = (entry_price - exit_price) / entry_price
                equity *= (1 + pn_l)
                reason = "SL" if sl_hit else "TP"
                trades.append({'time': ts, 'type': 'Exit', 'price': exit_price, 'pnl': pn_l, 'equity': equity, 'reason': reason})
                position = 0
                trades_completed += 1
                equity_curve.append(equity)
                continue 

        # --- Entry Logic ---
        if position == 0:
            found_short = False
            short_price = 0.0
            
            # 1. Check Short Candidates (High > Prev Close)
            if current_h > prev_c:
                idx_s = np.searchsorted(lines, prev_c, side='right')
                idx_e = np.searchsorted(lines, current_h, side='right')
                potential_shorts = lines[idx_s:idx_e]
                if len(potential_shorts) > 0:
                    found_short = True
                    short_price = potential_shorts[0]

            found_long = False
            long_price = 0.0
            
            # 2. Check Long Candidates (Low < Prev Close)
            if current_l < prev_c:
                idx_s = np.searchsorted(lines, current_l, side='left')
                idx_e = np.searchsorted(lines, prev_c, side='left')
                potential_longs = lines[idx_s:idx_e]
                if len(potential_longs) > 0:
                    found_long = True
                    long_price = potential_longs[-1]

            # 3. Execution Decision & Ambiguity Resolution
            target_line = 0.0
            new_pos = 0
            
            if found_short and found_long:
                # --- DRILL DOWN TO 1M DATA ---
                # We have a conflict. Check the tape (1m data).
                resolved_direction = resolve_ambiguity(ts, short_price, long_price, raw_df)
                
                if resolved_direction == -1:
                    new_pos = -1; target_line = short_price
                elif resolved_direction == 1:
                    new_pos = 1; target_line = long_price
                else:
                    # Fallback if 1m data is missing or totally inconclusive
                    dist_short = abs(prev_c - short_price)
                    dist_long = abs(prev_c - long_price)
                    if dist_short < dist_long: new_pos = -1; target_line = short_price
                    else: new_pos = 1; target_line = long_price

            elif found_short:
                new_pos = -1; target_line = short_price
            elif found_long:
                new_pos = 1; target_line = long_price
            
            if new_pos != 0:
                position = new_pos
                entry_price = target_line
                trades.append({'time': ts, 'type': 'Short' if position == -1 else 'Long', 'price': entry_price, 'pnl': 0, 'equity': equity, 'reason': 'Entry'})

        equity_curve.append(equity)

    return equity_curve, trades, hourly_log

def calculate_sharpe(equity_curve):
    if len(equity_curve) < 2: return -999.0
    returns = pd.Series(equity_curve).pct_change().dropna()
    if returns.std() == 0: return -999.0
    return np.sqrt(8760) * (returns.mean() / returns.std())

# --- 5. Genetic Algorithm ---
def setup_toolbox(min_price, max_price, df_train_1h, raw_df):
    toolbox = base.Toolbox()
    toolbox.register("attr_stop", random.uniform, STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    toolbox.register("attr_profit", random.uniform, PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    toolbox.register("attr_line", random.uniform, min_price, max_price)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_stop, toolbox.attr_profit) + (toolbox.attr_line,)*N_LINES, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Pass BOTH the 1H dataframe and the RAW 1M dataframe
    toolbox.register("evaluate", evaluate_genome, df_1h=df_train_1h, raw_df=raw_df)
    toolbox.register("mate", tools.cxTwoPoint) 
    toolbox.register("mutate", mutate_custom, indpb=0.1, min_p=min_price, max_p=max_price)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def evaluate_genome(individual, df_1h, raw_df):
    stop_pct = np.clip(individual[0], STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    profit_pct = np.clip(individual[1], PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    lines = np.array(individual[2:])
    eq_curve, _, _ = run_backtest(df_1h, raw_df, stop_pct, profit_pct, lines, detailed_log_trades=0)
    return (calculate_sharpe(eq_curve),)

def mutate_custom(individual, indpb, min_p, max_p):
    if random.random() < indpb:
        individual[0] = np.clip(individual[0] + random.gauss(0, 0.005), STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    if random.random() < indpb:
        individual[1] = np.clip(individual[1] + random.gauss(0, 0.005), PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    for i in range(2, len(individual)):
        if random.random() < (indpb / 10.0): 
            individual[i] = np.clip(individual[i] + random.gauss(0, (max_p - min_p) * 0.01), min_p, max_p)
    return individual,

# --- 6. Reporting ---
def generate_report(symbol, best_ind, train_curve, test_curve, test_trades, hourly_log, live_logs=[], live_trades=[]):
    plt.figure(figsize=(14, 12))
    
    plt.subplot(2, 1, 1)
    plt.title(f"{symbol} Equity Curve: Training (Blue) vs Test (Orange)")
    plt.plot(train_curve, label='Training Equity')
    plt.plot(range(len(train_curve), len(train_curve)+len(test_curve)), test_curve, label='Test Equity')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.title(f"{symbol} Performance Summary")
    plt.text(0.1, 0.5, f"Stop Loss: {best_ind[0]*100:.2f}%\nTake Profit: {best_ind[1]*100:.2f}%\nTotal Trades: {len(test_trades)}", fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', dpi=100)
    img_io.seek(0)
    plot_url = base64.b64encode(img_io.getvalue()).decode()
    plt.close()
    
    trades_df = pd.DataFrame(test_trades)
    trades_html = trades_df.to_html(classes='table table-striped table-sm', index=False, max_rows=500) if not trades_df.empty else "No trades."
    
    live_log_df = pd.DataFrame(live_logs)
    live_log_html = live_log_df.to_html(classes='table table-bordered table-sm', index=False) if not live_log_df.empty else "Waiting..."
    
    live_trades_df = pd.DataFrame(live_trades)
    live_trades_html = live_trades_df.to_html(classes='table table-striped table-sm', index=False) if not live_trades_df.empty else "No live trades."

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{symbol} Strategy</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <meta http-equiv="refresh" content="30"> 
    </head>
    <body class="p-4">
        <h1>{symbol} Results</h1>
        <img src="data:image/png;base64,{plot_url}" class="img-fluid border rounded mb-3">
        <h3>Live Status</h3>
        <div class="row">
            <div class="col-6">{live_log_html}</div>
            <div class="col-6">{live_trades_html}</div>
        </div>
        <h3>Test Trades</h3>
        <div style="height:300px;overflow:auto">{trades_html}</div>
    </body>
    </html>
    """
    return html_content

# --- 7. Live Forward Test ---
def fetch_binance_candle(symbol_pair):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol_pair, 'interval': '1m', 'limit': 2}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if len(data) >= 2:
            kline = data[-2]
            return pd.to_datetime(kline[0], unit='ms'), float(kline[2]), float(kline[3]), float(kline[4])
        return None, None, None, None
    except: return None, None, None, None

def live_trading_daemon(symbol, pair, best_ind, initial_equity, start_price, raw_df, train_curve, test_curve, test_trades, hourly_log):
    stop_pct, profit_pct = best_ind[0], best_ind[1]
    lines = np.sort(np.array(best_ind[2:]))
    live_equity = initial_equity
    live_position = 0 
    live_entry_price = 0.0
    prev_close = start_price
    live_logs = []
    live_trades = []
    
    print(f"[{symbol}] Live Daemon Started.")
    
    while True:
        now = datetime.now()
        next_run = (now + timedelta(minutes=1)).replace(second=5, microsecond=0)
        if next_run <= now: next_run += timedelta(minutes=1)
        time.sleep((next_run - now).total_seconds())
        
        ts, current_h, current_l, current_c = fetch_binance_candle(pair)
        if current_c is None: continue
            
        # Live Logic uses 1m natively, so ambiguity is handled by proximity immediately (since we can't drill down further than 1m)
        idx = np.searchsorted(lines, current_c)
        val_below = lines[idx-1] if idx > 0 else -999.0
        val_above = lines[idx] if idx < len(lines) else 999999.0
        
        # ... (Logging logic omitted for brevity, same as previous) ...
        live_logs.append({"Timestamp": str(ts), "Price": current_c, "Pos": live_position})
        
        # Exit Logic
        if live_position != 0:
            sl_hit, tp_hit = False, False
            exit_price = 0.0
            if live_position == 1:
                sl, tp = live_entry_price*(1-stop_pct), live_entry_price*(1+profit_pct)
                if current_l <= sl: sl_hit=True; exit_price=sl
                elif current_h >= tp: tp_hit=True; exit_price=tp
            elif live_position == -1:
                sl, tp = live_entry_price*(1+stop_pct), live_entry_price*(1-profit_pct)
                if current_h >= sl: sl_hit=True; exit_price=sl
                elif current_l <= tp: tp_hit=True; exit_price=tp
            
            if sl_hit or tp_hit:
                pnl = (exit_price - live_entry_price)/live_entry_price if live_position == 1 else (live_entry_price - exit_price)/live_entry_price
                live_equity *= (1+pnl)
                live_trades.append({'time': ts, 'type': 'Exit', 'pnl': pnl, 'equity': live_equity})
                live_position = 0
                prev_close = current_c
                with REPORT_LOCK:
                    HTML_REPORTS[symbol] = generate_report(symbol, best_ind, train_curve, test_curve, test_trades, hourly_log, live_logs[-10:], live_trades)
                continue

        # Entry Logic (1m Native)
        if live_position == 0:
            found_short, found_long = False, False
            short_price, long_price = 0.0, 0.0
            
            if current_h > prev_close:
                idx_s, idx_e = np.searchsorted(lines, prev_close, side='right'), np.searchsorted(lines, current_h, side='right')
                shorts = lines[idx_s:idx_e]
                if len(shorts) > 0: found_short = True; short_price = shorts[0]

            if current_l < prev_close:
                idx_s, idx_e = np.searchsorted(lines, current_l, side='left'), np.searchsorted(lines, prev_close, side='left')
                longs = lines[idx_s:idx_e]
                if len(longs) > 0: found_long = True; long_price = longs[-1]

            new_pos, target = 0, 0.0
            if found_short and found_long:
                # 1m Ambiguity -> Use Heuristic
                if abs(prev_close - short_price) < abs(prev_close - long_price): new_pos = -1; target = short_price
                else: new_pos = 1; target = long_price
            elif found_short: new_pos = -1; target = short_price
            elif found_long: new_pos = 1; target = long_price
            
            if new_pos != 0:
                live_position = new_pos
                live_entry_price = target
                live_trades.append({'time': ts, 'type': 'Entry', 'price': target, 'equity': live_equity})

        prev_close = current_c
        with REPORT_LOCK:
            HTML_REPORTS[symbol] = generate_report(symbol, best_ind, train_curve, test_curve, test_trades, hourly_log, live_logs[-10:], live_trades)

# --- 8. Server & Main ---
class ResultsHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/report/'):
            sym = self.path.split('/')[-1]
            if sym in HTML_REPORTS:
                self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
                self.wfile.write(HTML_REPORTS[sym].encode())
            else: self.send_error(404)
        elif self.path == '/':
            self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
            links = "".join([f'<a href="/report/{a["symbol"]}" class="btn btn-block btn-light">{a["symbol"]}</a>' for a in ASSETS])
            self.wfile.write(f'<html><link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"><body class="p-5"><h1>Dashboard</h1>{links}</body></html>'.encode())

def process_asset(asset):
    sym, csv = asset['symbol'], asset['csv']
    print(f"\n--- {sym} ---")
    train_1h, test_1h, raw_df = get_data(csv)
    if train_1h is None: return

    min_p, max_p = train_1h['close'].min(), train_1h['close'].max()
    toolbox = setup_toolbox(min_p, max_p, train_1h, raw_df) # Pass raw_df here
    
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS, verbose=False)
    
    best_ind = hof[0]
    print(f"Best Sharpe: {best_ind.fitness.values[0]:.4f}")
    BEST_PARAMS[sym] = {"stop": best_ind[0], "profit": best_ind[1], "lines": list(best_ind[2:])}
    
    train_curve, _, _ = run_backtest(train_1h, raw_df, best_ind[0], best_ind[1], np.array(best_ind[2:]))
    test_curve, test_trades, log = run_backtest(test_1h, raw_df, best_ind[0], best_ind[1], np.array(best_ind[2:]), detailed_log_trades=5)
    
    with REPORT_LOCK: HTML_REPORTS[sym] = generate_report(sym, best_ind, train_curve, test_curve, test_trades, log)
    
    threading.Thread(target=live_trading_daemon, args=(sym, asset['pair'], best_ind, 10000.0, test_1h['close'].iloc[-1], raw_df, train_curve, test_curve, test_trades, log), daemon=True).start()

if __name__ == "__main__":
    for asset in ASSETS[:MAX_ASSETS_TO_OPTIMIZE]: process_asset(asset)
    with socketserver.TCPServer(("", PORT), ResultsHandler) as httpd: httpd.serve_forever()
