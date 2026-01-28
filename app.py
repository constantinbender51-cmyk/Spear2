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
N_LINES = 200
POPULATION_SIZE = 40
GENERATIONS = 10
RISK_FREE_RATE = 0.0

# Ranges
STOP_PCT_RANGE = (0.001, 0.02)   # 0.1% to 2%
PROFIT_PCT_RANGE = (0.0004, 0.05) # 0.04% to 5%

warnings.filterwarnings("ignore")

# Asset List Mapping (Symbol -> Binance Pair & CSV Prefix)
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
# Key: Symbol (str) -> Value: HTML String
HTML_REPORTS = {} 
# Key: Symbol (str) -> Value: Dict of params
BEST_PARAMS = {}
# Lock for thread-safe updates to globals
REPORT_LOCK = threading.Lock()

# --- 1. DEAP Initialization (Global Scope) ---
# Must be done once globally, not inside loops/functions
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

        # Keeping 1H resampling for GA Optimization speed
        # The live bot will run on 1m, but we optimize on 1H to avoid 
        # waiting hours for the script to start.
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
        train = df_1h.iloc[:split_idx]
        test = df_1h.iloc[split_idx:]
        
        return train, test

    except Exception as e:
        print(f"CRITICAL DATA ERROR for {csv_filename}: {e}")
        return None, None

# --- 3. Strategy Logic ---
def run_backtest(df, stop_pct, profit_pct, lines, detailed_log_trades=0):
    closes = df['close'].values
    times = df.index
    
    equity = 10000.0
    equity_curve = [equity]
    position = 0          # 0: Flat, 1: Long, -1: Short
    entry_price = 0.0
    
    trades = []
    hourly_log = []
    
    lines = np.sort(lines)
    trades_completed = 0
    
    for i in range(1, len(df)):
        current_c = closes[i]
        prev_c = closes[i-1]
        ts = times[i]
        
        # --- Detailed Logging ---
        if detailed_log_trades > 0 and trades_completed < detailed_log_trades:
            idx = np.searchsorted(lines, current_c)
            val_below = lines[idx-1] if idx > 0 else -999.0
            val_above = lines[idx] if idx < len(lines) else 999999.0
            
            act_sl = np.nan
            act_tp = np.nan
            pos_str = "FLAT"
            
            if position == 1:
                pos_str = "LONG"
                act_sl = entry_price * (1 - stop_pct)
                act_tp = entry_price * (1 + profit_pct)
            elif position == -1:
                pos_str = "SHORT"
                act_sl = entry_price * (1 + stop_pct)
                act_tp = entry_price * (1 - profit_pct)
            
            log_entry = {
                "Timestamp": str(ts),
                "Price": f"{current_c:.2f}",
                "Nearest Below": f"{val_below:.2f}" if val_below != -999 else "None",
                "Nearest Above": f"{val_above:.2f}" if val_above != 999999 else "None",
                "Position": pos_str,
                "Active SL": f"{act_sl:.2f}" if not np.isnan(act_sl) else "-",
                "Active TP": f"{act_tp:.2f}" if not np.isnan(act_tp) else "-",
                "Equity": f"{equity:.2f}"
            }
            hourly_log.append(log_entry)

        # --- Strategy Execution ---
        if position != 0:
            sl_hit = False
            tp_hit = False
            exit_price = 0.0
            reason = ""

            if position == 1: # Long Logic
                sl_price = entry_price * (1 - stop_pct)
                tp_price = entry_price * (1 + profit_pct)
                if current_c <= sl_price:
                    sl_hit = True; exit_price = sl_price 
                elif current_c >= tp_price:
                    tp_hit = True; exit_price = tp_price

            elif position == -1: # Short Logic
                sl_price = entry_price * (1 + stop_pct)
                tp_price = entry_price * (1 - profit_pct)
                if current_c >= sl_price:
                    sl_hit = True; exit_price = sl_price
                elif current_c <= tp_price:
                    tp_hit = True; exit_price = tp_price
            
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

        if position == 0:
            p_min = min(prev_c, current_c)
            p_max = max(prev_c, current_c)
            
            idx_start = np.searchsorted(lines, p_min, side='right')
            idx_end = np.searchsorted(lines, p_max, side='right')
            
            crossed_lines = lines[idx_start:idx_end]
            
            if len(crossed_lines) > 0:
                target_line = 0.0
                new_pos = 0
                
                if current_c > prev_c: 
                    target_line = crossed_lines[0]
                    new_pos = -1
                elif current_c < prev_c:
                    target_line = crossed_lines[-1]
                    new_pos = 1
                
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

# --- 4. Genetic Algorithm ---
def setup_toolbox(min_price, max_price, df_train):
    toolbox = base.Toolbox()
    toolbox.register("attr_stop", random.uniform, STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    toolbox.register("attr_profit", random.uniform, PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    toolbox.register("attr_line", random.uniform, min_price, max_price)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_stop, toolbox.attr_profit) + (toolbox.attr_line,)*N_LINES, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_genome, df_train=df_train)
    toolbox.register("mate", tools.cxTwoPoint) 
    toolbox.register("mutate", mutate_custom, indpb=0.1, min_p=min_price, max_p=max_price)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def evaluate_genome(individual, df_train):
    stop_pct = np.clip(individual[0], STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    profit_pct = np.clip(individual[1], PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    lines = np.array(individual[2:])
    eq_curve, _, _ = run_backtest(df_train, stop_pct, profit_pct, lines, detailed_log_trades=0)
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

# --- 5. Reporting ---
def generate_report(symbol, best_ind, train_data, test_data, train_curve, test_curve, test_trades, hourly_log, live_logs=[], live_trades=[]):
    plt.figure(figsize=(14, 12))
    
    # 1. Equity Curve
    plt.subplot(2, 1, 1)
    plt.title(f"{symbol} Equity Curve: Training (Blue) vs Test (Orange)")
    plt.plot(train_curve, label='Training Equity')
    plt.plot(range(len(train_curve), len(train_curve)+len(test_curve)), test_curve, label='Test Equity')
    plt.legend()
    plt.grid(True)
    
    # 2. Full Price Action
    plt.subplot(2, 1, 2)
    plt.title(f"{symbol} Test Set Price Action & Grid Lines")
    plt.plot(test_data.index, test_data['close'], color='black', alpha=0.6, label='Price', linewidth=0.8)
    
    lines = best_ind[2:]
    min_test = test_data['low'].min()
    max_test = test_data['high'].max()
    margin = (max_test - min_test) * 0.1
    visible_lines = [l for l in lines if (min_test - margin) < l < (max_test + margin)]
    
    for l in visible_lines:
        plt.axhline(y=l, color='blue', alpha=0.1, linewidth=0.5)
    
    plt.tight_layout()
    
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', dpi=100)
    img_io.seek(0)
    plot_url = base64.b64encode(img_io.getvalue()).decode()
    plt.close()
    
    trades_df = pd.DataFrame(test_trades)
    trades_html = trades_df.to_html(classes='table table-striped table-sm', index=False, max_rows=500) if not trades_df.empty else "No trades."
    
    hourly_df = pd.DataFrame(hourly_log)
    hourly_html = hourly_df.to_html(classes='table table-bordered table-sm table-hover', index=False) if not hourly_df.empty else "No hourly data recorded."

    live_log_df = pd.DataFrame(live_logs)
    live_log_html = live_log_df.to_html(classes='table table-bordered table-sm table-hover', index=False) if not live_log_df.empty else "Waiting for next minute trigger..."
    
    live_trades_df = pd.DataFrame(live_trades)
    live_trades_html = live_trades_df.to_html(classes='table table-striped table-sm', index=False) if not live_trades_df.empty else "No live trades yet."

    params_html = f"""
    <ul class="list-group">
        <li class="list-group-item"><strong>Stop Loss:</strong> {best_ind[0]*100:.4f}%</li>
        <li class="list-group-item"><strong>Take Profit:</strong> {best_ind[1]*100:.4f}%</li>
        <li class="list-group-item"><strong>Active Grid Lines:</strong> {N_LINES}</li>
        <li class="list-group-item"><a href="/api/parameters?symbol={symbol}" target="_blank">View JSON Parameters</a></li>
    </ul>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{symbol} Strategy Results</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <meta http-equiv="refresh" content="30"> 
        <style>body {{ padding: 20px; }} h3 {{ margin-top: 30px; }} th {{ position: sticky; top: 0; background: white; }}</style>
    </head>
    <body>
        <div class="container-fluid">
            <a href="/" class="btn btn-secondary mb-3">&larr; Back to Dashboard</a>
            <h1 class="mb-4">{symbol} Grid Strategy GA Results</h1>
            <div class="row">
                <div class="col-md-4">{params_html}</div>
                <div class="col-md-8 text-right">
                    <h5>Test Sharpe: {calculate_sharpe(test_curve):.4f}</h5>
                </div>
            </div>
            <hr>
            <h3>Performance Charts</h3>
            <img src="data:image/png;base64,{plot_url}" class="img-fluid border rounded">
            
            <hr>
            <div id="live-section" style="background-color: #f8f9fa; padding: 15px; border-left: 5px solid #28a745;">
                <h2 class="text-success">{symbol} Live Forward Test (Binance 1m)</h2>
                <p><strong>Status:</strong> Running. Fetches candle at XX:XX:05 (Every Minute).</p>
                <div class="row">
                    <div class="col-md-6">
                        <h4>Live Minute State</h4>
                        <div style="max-height: 400px; overflow-y: scroll; border: 1px solid #ddd; background: white;">
                            {live_log_html}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h4>Live Trade Log</h4>
                         <div style="max-height: 400px; overflow-y: scroll; border: 1px solid #ddd; background: white;">
                            {live_trades_html}
                        </div>
                    </div>
                </div>
            </div>

            <hr>
            <h3>Trade Log (Test Set)</h3>
            <div style="max-height: 400px; overflow-y: scroll; border: 1px solid #ddd;">{trades_html}</div>
            
            <hr>
            <h3>Hourly Details (First 5 Trades Timeline)</h3>
            <div style="max-height: 600px; overflow-y: scroll; border: 1px solid #ddd;">
                {hourly_html}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# --- 6. Live Forward Test Logic (1 Minute Update) ---
def fetch_binance_candle(symbol_pair):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol_pair,
            'interval': '1m', # CHANGED TO 1 MINUTE
            'limit': 2 
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if len(data) >= 2:
            kline = data[-2] 
            ts = pd.to_datetime(kline[0], unit='ms')
            close_price = float(kline[4])
            return ts, close_price
        return None, None
    except Exception as e:
        print(f"[{symbol_pair}] Binance API Error: {e}")
        return None, None

def live_trading_daemon(symbol, pair, best_ind, initial_equity, start_price, train_df, test_df, train_curve, test_curve, test_trades, hourly_log):
    
    stop_pct = best_ind[0]
    profit_pct = best_ind[1]
    lines = np.sort(np.array(best_ind[2:]))
    
    live_equity = initial_equity
    live_position = 0 
    live_entry_price = 0.0
    prev_close = start_price
    
    live_logs = []
    live_trades = []
    
    # Stagger start to avoid rate limits
    time.sleep(random.uniform(1, 10))
    print(f"[{symbol}] Live Trading Daemon Started (1m interval).")
    
    while True:
        now = datetime.now()
        # Sleep until next MINUTE + 5 seconds
        next_run = (now + timedelta(minutes=1)).replace(second=5, microsecond=0)
        
        if next_run <= now:
            next_run += timedelta(minutes=1)
            
        sleep_sec = (next_run - now).total_seconds()
        # Add tiny random jitter to avoid exact sync
        sleep_sec += random.uniform(0.1, 1.0)
        
        time.sleep(sleep_sec)
        
        ts, current_c = fetch_binance_candle(pair)
        
        if current_c is None:
            print(f"[{symbol}] Failed to fetch data. Skipping.")
            continue
            
        print(f"[{symbol}] Processing {ts} Close: {current_c}")
        
        idx = np.searchsorted(lines, current_c)
        val_below = lines[idx-1] if idx > 0 else -999.0
        val_above = lines[idx] if idx < len(lines) else 999999.0
        
        act_sl = np.nan
        act_tp = np.nan
        pos_str = "FLAT"
        
        if live_position == 1:
            pos_str = "LONG"
            act_sl = live_entry_price * (1 - stop_pct)
            act_tp = live_entry_price * (1 + profit_pct)
        elif live_position == -1:
            pos_str = "SHORT"
            act_sl = live_entry_price * (1 + stop_pct)
            act_tp = live_entry_price * (1 - profit_pct)
            
        log_entry = {
            "Timestamp": str(ts),
            "Price": f"{current_c:.2f}",
            "Nearest Below": f"{val_below:.2f}" if val_below != -999 else "None",
            "Nearest Above": f"{val_above:.2f}" if val_above != 999999 else "None",
            "Position": pos_str,
            "Active SL": f"{act_sl:.2f}" if not np.isnan(act_sl) else "-",
            "Active TP": f"{act_tp:.2f}" if not np.isnan(act_tp) else "-",
            "Equity": f"{live_equity:.2f}"
        }
        live_logs.append(log_entry)
        
        if live_position != 0:
            sl_hit = False
            tp_hit = False
            exit_price = 0.0
            reason = ""

            if live_position == 1:
                sl_price = live_entry_price * (1 - stop_pct)
                tp_price = live_entry_price * (1 + profit_pct)
                if current_c <= sl_price:
                    sl_hit = True; exit_price = sl_price
                elif current_c >= tp_price:
                    tp_hit = True; exit_price = tp_price
            elif live_position == -1:
                sl_price = live_entry_price * (1 + stop_pct)
                tp_price = live_entry_price * (1 - profit_pct)
                if current_c >= sl_price:
                    sl_hit = True; exit_price = sl_price
                elif current_c <= tp_price:
                    tp_hit = True; exit_price = tp_price
            
            if sl_hit or tp_hit:
                if live_position == 1: pn_l = (exit_price - live_entry_price) / live_entry_price
                else: pn_l = (live_entry_price - exit_price) / live_entry_price
                
                live_equity *= (1 + pn_l)
                reason = "SL" if sl_hit else "TP"
                live_trades.append({'time': ts, 'type': 'Exit', 'price': exit_price, 'pnl': pn_l, 'equity': live_equity, 'reason': reason})
                live_position = 0
                
                prev_close = current_c
                with REPORT_LOCK:
                    HTML_REPORTS[symbol] = generate_report(symbol, best_ind, train_df, test_df, train_curve, test_curve, test_trades, hourly_log, live_logs, live_trades)
                continue

        if live_position == 0:
            p_min = min(prev_close, current_c)
            p_max = max(prev_close, current_c)
            
            idx_start = np.searchsorted(lines, p_min, side='right')
            idx_end = np.searchsorted(lines, p_max, side='right')
            crossed_lines = lines[idx_start:idx_end]
            
            if len(crossed_lines) > 0:
                target_line = 0.0
                new_pos = 0
                if current_c > prev_close:
                    target_line = crossed_lines[0]
                    new_pos = -1
                elif current_c < prev_close:
                    target_line = crossed_lines[-1]
                    new_pos = 1
                
                if new_pos != 0:
                    live_position = new_pos
                    live_entry_price = target_line
                    live_trades.append({'time': ts, 'type': 'Short' if live_position == -1 else 'Long', 'price': live_entry_price, 'pnl': 0, 'equity': live_equity, 'reason': 'Entry'})

        prev_close = current_c
        with REPORT_LOCK:
            HTML_REPORTS[symbol] = generate_report(symbol, best_ind, train_df, test_df, train_curve, test_curve, test_trades, hourly_log, live_logs, live_trades)

# --- 7. Server Handler ---
class ResultsHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)

        if path == '/api/parameters':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            symbol = query.get('symbol', [None])[0]
            if symbol and symbol in BEST_PARAMS:
                self.wfile.write(json.dumps(BEST_PARAMS[symbol]).encode('utf-8'))
            else:
                self.wfile.write(json.dumps(BEST_PARAMS).encode('utf-8'))
                
        elif path.startswith('/report/'):
            symbol = path.split('/')[-1]
            if symbol in HTML_REPORTS:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                with REPORT_LOCK:
                    self.wfile.write(HTML_REPORTS[symbol].encode('utf-8'))
            else:
                self.send_error(404, "Report not found for symbol")
                
        elif path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Dashboard Index
            links = ""
            for asset in ASSETS:
                sym = asset['symbol']
                if sym in HTML_REPORTS:
                    links += f'<a href="/report/{sym}" class="list-group-item list-group-item-action">{sym} Strategy Report</a>'
                else:
                    links += f'<div class="list-group-item list-group-item-light">{sym} (Initializing...)</div>'
            
            dashboard = f"""
            <html>
            <head>
                <title>Multi-Asset Grid Bot</title>
                <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
                <meta http-equiv="refresh" content="30">
            </head>
            <body class="p-5">
                <h1>Active Grid Strategies</h1>
                <div class="list-group mt-4">
                    {links}
                </div>
            </body>
            </html>
            """
            self.wfile.write(dashboard.encode('utf-8'))
        else:
            self.send_error(404)

# --- 8. Main Execution Loop ---
def process_asset(asset_config):
    sym = asset_config['symbol']
    csv = asset_config['csv']
    pair = asset_config['pair']
    
    print(f"\n--- Starting Optimization for {sym} ---")
    
    # 1. Get Data
    train_df, test_df = get_data(csv)
    if train_df is None:
        print(f"Skipping {sym} due to data error.")
        return

    # 2. Setup GA
    min_p, max_p = train_df['close'].min(), train_df['close'].max()
    toolbox = setup_toolbox(min_p, max_p, train_df)

    # 3. Run GA
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    print(f"[{sym}] Evolving...")
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=False)
    
    best_ind = hof[0]
    print(f"[{sym}] Best Sharpe: {best_ind.fitness.values[0]:.4f}")

    # 4. Save Params
    BEST_PARAMS[sym] = {
        "stop_percent": best_ind[0],
        "profit_percent": best_ind[1],
        "line_prices": list(best_ind[2:])
    }

    # 5. Final Tests
    train_curve, _, _ = run_backtest(train_df, best_ind[0], best_ind[1], np.array(best_ind[2:]), detailed_log_trades=0)
    test_curve, test_trades, hourly_log = run_backtest(test_df, best_ind[0], best_ind[1], np.array(best_ind[2:]), detailed_log_trades=5)
    
    # 6. Generate Initial Report
    with REPORT_LOCK:
        HTML_REPORTS[sym] = generate_report(sym, best_ind, train_df, test_df, train_curve, test_curve, test_trades, hourly_log)

    # 7. Start Live Thread
    last_test_close = test_df['close'].iloc[-1]
    t = threading.Thread(
        target=live_trading_daemon, 
        args=(sym, pair, best_ind, 10000.0, last_test_close, train_df, test_df, train_curve, test_curve, test_trades, hourly_log),
        daemon=True
    )
    t.start()
    print(f"[{sym}] Live thread launched.")

if __name__ == "__main__":
    print("Initializing Multi-Asset Grid System...")
    
    # Process assets sequentially for GA to avoid massive CPU contention,
    # then spawn threads for live waiting.
    for asset in ASSETS:
        process_asset(asset)
    
    print("\nAll assets processed. Starting Web Server...")
    print(f"Serving Dashboard at http://localhost:{PORT}/")
    
    with socketserver.TCPServer(("", PORT), ResultsHandler) as httpd:
        try: httpd.serve_forever()
        except KeyboardInterrupt: httpd.server_close()
