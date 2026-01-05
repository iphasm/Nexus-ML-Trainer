"""
ML Model Training Script v3.2 - Standalone Cloud Trainer
Designed to run on Railway as a separate service from the main bot.

Features:
- XGBoost Classifier with proper regularization
- RobustScaler for crypto outlier handling
- TimeSeriesSplit for chronological validation
- Automatic model upload to PostgreSQL
"""

import asyncio
import os
import sys
import time
import signal
import logging
import argparse
from datetime import datetime
from tqdm import tqdm

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
from binance.client import Client
import warnings
warnings.filterwarnings('ignore')

from indicators import add_indicators, FEATURE_COLUMNS
from model_uploader import upload_model, init_ml_models_table

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global flag for interruption handling
interrupted = False

# Default symbols to train on (can be overridden via config)
DEFAULT_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 'APTUSDT'
]

INTERVAL = '15m'

# Strategy SL/TP configurations (matching real trading logic)
STRATEGY_PARAMS = {
    'trend': {'sl_pct': 0.02, 'tp_pct': 0.04, 'min_adx': 25},
    'scalp': {'sl_pct': 0.008, 'tp_pct': 0.012, 'min_atr_pct': 1.5},
    'grid': {'sl_pct': 0.015, 'tp_pct': 0.015, 'max_atr_pct': 0.8},
    'mean_rev': {'sl_pct': 0.018, 'tp_pct': 0.025, 'rsi_low': 30, 'rsi_high': 70},
}


def signal_handler(signum, frame):
    """Handle Ctrl+C interruption gracefully"""
    global interrupted
    interrupted = True
    logger.warning("⚠️ Interrupción detectada (Ctrl+C). Finalizando...")


def fetch_crypto_data(symbol: str, max_candles: int = 15000, verbose: bool = False) -> pd.DataFrame:
    """
    Fetch crypto data with fallback sources.
    Primary: Binance Futures API (with API keys and proxy if available)
    Fallback: yfinance (for BTC, ETH, etc.)
    """
    global interrupted
    
    if interrupted:
        return None
    
    # Get API credentials and proxy from environment
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    proxy_url = os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY') or os.getenv('PROXY_URL')
    
    # Try Binance first
    try:
        if verbose:
            has_keys = "con API keys" if api_key else "sin API keys"
            has_proxy = "con proxy" if proxy_url else "sin proxy"
            print(f"  📊 Intentando Binance para {symbol} ({has_keys}, {has_proxy})...", flush=True)
        
        # Configure client with credentials and proxy
        client_kwargs = {}
        if api_key and api_secret:
            client_kwargs['api_key'] = api_key
            client_kwargs['api_secret'] = api_secret
        
        if proxy_url:
            client_kwargs['requests_params'] = {
                'proxies': {
                    'http': proxy_url,
                    'https': proxy_url
                }
            }
        
        client = Client(**client_kwargs)
        
        # Fetch up to 1500 candles per request (Binance limit)
        klines = client.futures_klines(
            symbol=symbol,
            interval=INTERVAL,
            limit=min(max_candles, 1500)
        )
        
        if klines and len(klines) > 0:
            if verbose:
                print(f"  ✅ Recibidos {len(klines)} registros de Binance", flush=True)
            
            # Process data
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df = df.astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
        else:
            logger.warning(f"Binance returned empty data for {symbol}")
            
    except Exception as e:
        logger.warning(f"Binance failed for {symbol}: {str(e)[:100]}")
    
    # Fallback to yfinance
    try:
        import yfinance as yf
        
        # Map symbol to yfinance ticker
        yf_symbol_map = {
            'BTCUSDT': 'BTC-USD',
            'ETHUSDT': 'ETH-USD',
            'BNBUSDT': 'BNB-USD',
            'SOLUSDT': 'SOL-USD',
            'XRPUSDT': 'XRP-USD',
            'ADAUSDT': 'ADA-USD',
            'AVAXUSDT': 'AVAX-USD',
            'DOGEUSDT': 'DOGE-USD',
            'DOTUSDT': 'DOT-USD',
            'MATICUSDT': 'MATIC-USD',
            'LINKUSDT': 'LINK-USD',
            'ATOMUSDT': 'ATOM-USD',
            'LTCUSDT': 'LTC-USD',
            'UNIUSDT': 'UNI-USD',
            'APTUSDT': 'APT-USD',
        }
        
        yf_symbol = yf_symbol_map.get(symbol)
        if not yf_symbol:
            logger.warning(f"No yfinance mapping for {symbol}")
            return None
        
        if verbose:
            print(f"  🔄 Intentando yfinance para {yf_symbol}...", flush=True)
        
        # Download 30 days of 15m data (max available from yfinance)
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="30d", interval="15m")
        
        if df.empty:
            logger.warning(f"yfinance returned empty data for {yf_symbol}")
            return None
        
        # Rename columns to match expected format
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        # Rename 'datetime' or 'date' to 'timestamp'
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        
        # Ensure required columns exist
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            logger.warning(f"yfinance missing columns for {yf_symbol}")
            return None
        
        df = df[required].copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if verbose:
            print(f"  ✅ Recibidos {len(df)} registros de yfinance", flush=True)
        
        return df
        
    except Exception as e:
        logger.error(f"yfinance also failed for {symbol}: {str(e)[:100]}")
        return None



def simulate_trade(df: pd.DataFrame, idx: int, strategy: str, lookforward: int = 24) -> bool:
    """
    Simulate a trade starting at index 'idx' using strategy parameters.
    Returns True if TP was hit before SL (profitable), False otherwise.
    """
    params = STRATEGY_PARAMS.get(strategy, STRATEGY_PARAMS['mean_rev'])
    entry_price = df.iloc[idx]['close']
    sl_pct = params['sl_pct']
    tp_pct = params['tp_pct']
    
    # Determine direction based on indicators
    rsi = df.iloc[idx]['rsi']
    trend = df.iloc[idx]['trend_str']
    
    # Simple direction logic
    if strategy == 'trend':
        is_long = trend > 0
    elif strategy == 'mean_rev':
        is_long = rsi < 50
    else:
        is_long = rsi < 50
    
    # Set SL/TP prices
    if is_long:
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)
    else:
        sl_price = entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 - tp_pct)
    
    # Simulate forward
    max_idx = min(idx + lookforward, len(df) - 1)
    
    for i in range(idx + 1, max_idx + 1):
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']
        
        if is_long:
            if low <= sl_price:
                return False
            if high >= tp_price:
                return True
        else:
            if high >= sl_price:
                return False
            if low <= tp_price:
                return True
    
    # Neither hit - check if in profit at end
    final_price = df.iloc[max_idx]['close']
    if is_long:
        return final_price > entry_price
    else:
        return final_price < entry_price


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each row with the best strategy based on trade simulation.
    """
    n = len(df)
    labels = ['mean_rev'] * n
    
    adx = df['adx'].values
    atr_pct = df['atr_pct'].values
    rsi = df['rsi'].values
    
    for idx in range(n - 25):
        eligible = []
        
        if adx[idx] > 25:
            eligible.append('trend')
        if atr_pct[idx] > 1.5:
            eligible.append('scalp')
        if atr_pct[idx] < 0.8:
            eligible.append('grid')
        if rsi[idx] < 35 or rsi[idx] > 65:
            eligible.append('mean_rev')
        
        if not eligible:
            eligible = ['mean_rev']
        
        best_strategy = 'mean_rev'
        for strat in eligible:
            if simulate_trade(df, idx, strat):
                best_strategy = strat
                break
        
        labels[idx] = best_strategy
    
    df['target'] = labels
    df = df.iloc[:-25].copy()
    df.dropna(inplace=True)
    
    return df


def train(symbols: list = None, max_candles: int = 15000, verbose: bool = False):
    """Main training function."""
    # ANSI Colors
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"
    
    signal.signal(signal.SIGINT, signal_handler)
    
    total_start_time = time.time()
    symbols = symbols or DEFAULT_SYMBOLS
    
    print("=" * 70, flush=True)
    print("🧠 NEXUS CORTEX CLOUD TRAINER v3.2", flush=True)
    print("=" * 70, flush=True)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"📊 Symbols: {len(symbols)}", flush=True)
    print(f"🕯️ Candles per symbol: {max_candles:,}", flush=True)
    print("", flush=True)
    
    # Initialize database table
    try:
        init_ml_models_table()
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        print(f"{RED}❌ Cannot connect to database. Check DATABASE_URL.{RESET}")
        return
    
    # Collect data
    all_data = []
    successful_downloads = 0
    
    with tqdm(total=len(symbols), desc="📥 Descargando Datos", unit="sym") as pbar:
        for symbol in symbols:
            if interrupted:
                break
            
            try:
                df = fetch_crypto_data(symbol, max_candles, verbose)
                
                if df is not None and not df.empty:
                    df = add_indicators(df)
                    df = label_data(df)
                    
                    if len(df) > 100:
                        all_data.append(df)
                        successful_downloads += 1
                        pbar.set_postfix_str(f"{GREEN}✓ {symbol}{RESET}")
                    else:
                        pbar.set_postfix_str(f"{YELLOW}⚠ {symbol}{RESET}")
                else:
                    pbar.set_postfix_str(f"{RED}✗ {symbol}{RESET}")
                    
            except Exception as e:
                pbar.set_postfix_str(f"{RED}💥 {symbol}{RESET}")
                logger.error(f"Error processing {symbol}: {e}")
            
            pbar.update(1)
    
    if not all_data:
        print(f"{RED}❌ No data collected. Aborting.{RESET}")
        return
    
    print(f"\n✅ Data collected: {successful_downloads}/{len(symbols)} symbols")
    
    # Prepare training data
    full_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 Total samples: {len(full_df):,}")
    
    X = full_df[FEATURE_COLUMNS]
    y = full_df['target']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    print(f"🎯 Classes: {', '.join(class_names)}")
    print(f"\n📈 Class distribution:")
    for label, count in y.value_counts().items():
        pct = count / len(y) * 100
        print(f"   • {label:8}: {count:>8,} ({pct:5.1f}%)")
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute sample weights
    sample_weights = compute_sample_weight('balanced', y_encoded)
    
    # Cross-validation
    print(f"\n🔍 Running 5-fold TimeSeriesSplit cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        weights = sample_weights[train_idx]
        
        cv_model = XGBClassifier(
            objective='multi:softprob',
            num_class=len(class_names),
            max_depth=5,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        cv_model.fit(X_train, y_train, sample_weight=weights)
        score = cv_model.score(X_val, y_val)
        cv_scores.append(score)
        print(f"   Fold {fold+1}: {score:.3f}")
    
    cv_scores = np.array(cv_scores)
    print(f"\n📊 CV Results: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Final training
    print(f"\n🏋️ Training final model on full dataset...")
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(class_names),
        max_depth=5,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_scaled, y_encoded, sample_weight=sample_weights)
    
    # Evaluate on last 20%
    split_idx = int(len(X_scaled) * 0.8)
    X_test = X_scaled[split_idx:]
    y_test = y_encoded[split_idx:]
    
    preds = model.predict(X_test)
    test_accuracy = (preds == y_test).mean()
    
    print(f"\n📈 TEST SET EVALUATION:")
    print(classification_report(y_test, preds, target_names=class_names, 
                                labels=range(len(class_names)), zero_division=0))
    
    # Feature importance
    print(f"\n🔑 TOP 10 FEATURES:")
    importance_pairs = sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1])
    for i, (feat, imp) in enumerate(importance_pairs[:10], 1):
        print(f"   {i:2d}. {feat:20} {imp:.3f}")
    
    # Upload model to PostgreSQL
    print(f"\n💾 Uploading model to PostgreSQL...")
    
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': FEATURE_COLUMNS
    }
    
    version = f"v3.2-{datetime.now().strftime('%Y%m%d-%H%M')}"
    
    metadata = {
        'symbols': symbols,
        'candles_per_symbol': max_candles,
        'total_samples': len(full_df),
        'training_time_seconds': time.time() - total_start_time,
        'class_distribution': {str(k): int(v) for k, v in y.value_counts().items()}
    }
    
    success = upload_model(
        model_data=model_data,
        scaler=scaler,
        version=version,
        accuracy=test_accuracy,
        cv_score=cv_scores.mean(),
        feature_names=FEATURE_COLUMNS,
        metadata=metadata
    )
    
    if success:
        print(f"{GREEN}✅ Model uploaded successfully: {version}{RESET}")
    else:
        print(f"{RED}❌ Model upload failed!{RESET}")
    
    # Summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*70}")
    print(f"🎉 TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"⏱️ Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"📊 Samples: {len(full_df):,}")
    print(f"🎯 Test Accuracy: {test_accuracy:.3f}")
    print(f"📈 CV Score: {cv_scores.mean():.3f}")
    print(f"📦 Model version: {version}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Nexus Cortex Cloud Trainer')
    parser.add_argument('--candles', type=int, default=15000,
                       help='Number of 15m candles to analyze')
    parser.add_argument('--symbols', type=int, default=None,
                       help='Limit number of symbols (for testing)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--auto', action='store_true',
                       help='Auto mode (no input prompts)')
    
    args = parser.parse_args()
    
    symbols = DEFAULT_SYMBOLS
    if args.symbols:
        symbols = symbols[:args.symbols]
    
    try:
        train(symbols=symbols, max_candles=args.candles, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n👋 Training cancelled by user.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not args.auto:
            input("\n🔴 Press ENTER to exit...")
