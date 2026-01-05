"""
Technical Indicators Module
Calculates all indicators required for ML model training.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate proper ADX using pandas-ta."""
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=period)
    if adx_df is None:
        return pd.Series(0, index=df.index)
    return adx_df.iloc[:, 0].fillna(0).clip(0, 100)


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate MFI using pandas-ta."""
    mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=period)
    if mfi is None:
        return pd.Series(50, index=df.index)
    return mfi.fillna(50).clip(0, 100)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ALL technical indicators for training.
    EXTENDED FEATURE SET for v3.1
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # === BASIC INDICATORS (using pandas-ta) ===
    # RSI (14)
    df['rsi'] = ta.rsi(close, length=14)
    if df['rsi'] is None:
        df['rsi'] = 50
    df['rsi'] = df['rsi'].fillna(50)
    
    # ATR (14)
    df['atr'] = ta.atr(high, low, close, length=14)
    if df['atr'] is None:
        df['atr'] = 0
    df['atr'] = df['atr'].fillna(0)
    df['atr_pct'] = (df['atr'] / close) * 100
    
    # ADX
    df['adx'] = calculate_adx(df, period=14)
    
    # EMAs (using pandas-ta)
    df['ema_9'] = ta.ema(close, length=9)
    df['ema_20'] = ta.ema(close, length=20)
    df['ema_50'] = ta.ema(close, length=50)
    df['ema_200'] = ta.ema(close, length=200)
    
    # Fill NaN in EMAs
    for col in ['ema_9', 'ema_20', 'ema_50', 'ema_200']:
        if df[col] is None:
            df[col] = close
        df[col] = df[col].fillna(method='bfill').fillna(close)
    
    # Trend Strength (EMA divergence)
    df['trend_str'] = (df['ema_20'] - df['ema_50']) / close * 100
    
    # Volume Change
    df['vol_ma_5'] = volume.rolling(5).mean()
    df['vol_ma_20'] = volume.rolling(20).mean()
    df['vol_change'] = (df['vol_ma_5'] - df['vol_ma_20']) / (df['vol_ma_20'] + 1e-10)
    
    # === v3.0 FEATURES ===
    
    # MACD (using pandas-ta)
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        df['macd'] = macd_df.iloc[:, 0]
        df['macd_signal'] = macd_df.iloc[:, 2]
        df['macd_hist'] = macd_df.iloc[:, 1]
    else:
        df['macd'] = 0
        df['macd_signal'] = 0
        df['macd_hist'] = 0
    df['macd_hist_norm'] = df['macd_hist'] / close * 100
    
    # Bollinger Bands (using pandas-ta)
    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None:
        df['bb_middle'] = bb.iloc[:, 1]
        df['bb_upper'] = bb.iloc[:, 2]
        df['bb_lower'] = bb.iloc[:, 0]
    else:
        df['bb_middle'] = close
        df['bb_upper'] = close
        df['bb_lower'] = close
    df['bb_std'] = (df['bb_upper'] - df['bb_middle']) / 2
    df['bb_width'] = (df['bb_std'] * 2) / (df['bb_middle'] + 1e-10) * 100
    df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # Price Momentum (Rate of Change)
    df['roc_5'] = (close - close.shift(5)) / close.shift(5) * 100
    df['roc_10'] = (close - close.shift(10)) / close.shift(10) * 100
    
    # OBV (On-Balance Volume) - normalized
    obv = (np.sign(close.diff()) * volume).cumsum()
    df['obv_change'] = obv.diff(5) / (obv.rolling(20).mean() + 1e-10)
    
    # Price position in range (0-1), using 20 period
    df['price_position'] = (close - low.rolling(20).min()) / (
        high.rolling(20).max() - low.rolling(20).min() + 1e-10
    )
    
    # Candle patterns (simple)
    df['body_pct'] = abs(close - df['open']) / (high - low + 1e-10)
    df['upper_wick'] = (high - pd.concat([close, df['open']], axis=1).max(axis=1)) / (high - low + 1e-10)
    df['lower_wick'] = (pd.concat([close, df['open']], axis=1).min(axis=1) - low) / (high - low + 1e-10)
    
    # Trend direction binary
    df['above_ema200'] = (close > df['ema_200']).astype(int)
    df['ema_cross'] = (df['ema_9'] > df['ema_20']).astype(int)
    
    # === NEW v3.1 FEATURES ===
    
    # EMA20 Slope (momentum direction) - change over 5 periods
    df['ema20_slope'] = (df['ema_20'] - df['ema_20'].shift(5)) / close * 100
    
    # MFI (Money Flow Index) - volume-weighted RSI alternative
    df['mfi'] = calculate_mfi(df, period=14)
    
    # Distance to 50-period High/Low (structure)
    high_50 = high.rolling(50).max()
    low_50 = low.rolling(50).min()
    df['dist_50_high'] = (close - high_50) / close * 100  # Negative = below high
    df['dist_50_low'] = (close - low_50) / close * 100    # Positive = above low
    
    # Time-based features (seasonality)
    if 'timestamp' in df.columns:
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    else:
        df['hour_of_day'] = 12  # Default to noon
        df['day_of_week'] = 2   # Default to Wednesday
    
    # === v3.2 ADDITIONAL FEATURES ===
    
    # Extended ROC
    df['roc_21'] = (close - close.shift(21)) / close.shift(21) * 100
    df['roc_50'] = (close - close.shift(50)) / close.shift(50) * 100
    
    # Williams %R
    williams = ta.willr(high, low, close, length=14)
    df['williams_r'] = williams if williams is not None else -50
    
    # CCI
    cci = ta.cci(high, low, close, length=20)
    df['cci'] = cci.fillna(0) if cci is not None else 0
    
    # Ultimate Oscillator
    uo = ta.uo(high, low, close)
    df['ultimate_osc'] = uo.fillna(50) if uo is not None else 50
    
    # Volume features
    df['volume_roc_5'] = (volume - volume.shift(5)) / (volume.shift(5) + 1e-10) * 100
    df['volume_roc_21'] = (volume - volume.shift(21)) / (volume.shift(21) + 1e-10) * 100
    
    # Chaikin Money Flow
    cmf = ta.cmf(high, low, close, volume, length=20)
    df['chaikin_mf'] = cmf.fillna(0) if cmf is not None else 0
    
    # Force Index
    df['force_index'] = close.diff() * volume
    df['force_index'] = df['force_index'].rolling(13).mean()
    
    # Ease of Movement
    eom = ta.eom(high, low, close, volume, length=14)
    df['ease_movement'] = eom.fillna(0) if eom is not None else 0
    
    # Structure features
    sma_20 = ta.sma(close, length=20)
    sma_50 = ta.sma(close, length=50)
    df['dist_sma20'] = (close - sma_20) / close * 100 if sma_20 is not None else 0
    df['dist_sma50'] = (close - sma_50) / close * 100 if sma_50 is not None else 0
    
    # Pivot distance (simplified)
    pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    df['pivot_dist'] = (close - pivot) / close * 100
    
    # Fib distance (from recent swing)
    recent_high = high.rolling(20).max()
    recent_low = low.rolling(20).min()
    fib_382 = recent_low + (recent_high - recent_low) * 0.382
    df['fib_dist'] = (close - fib_382) / close * 100
    
    # Volatility by session
    df['morning_volatility'] = df['atr_pct'].rolling(8).mean()  # ~2 hours
    df['afternoon_volatility'] = df['atr_pct'].rolling(16).mean()  # ~4 hours
    
    # Gap detection
    df['gap_up'] = ((df['open'] - close.shift(1)) / close.shift(1) * 100).clip(lower=0)
    df['gap_down'] = ((df['open'] - close.shift(1)) / close.shift(1) * 100).clip(upper=0).abs()
    
    # Range change
    df['range_change'] = ((high - low) - (high.shift(1) - low.shift(1))) / (high.shift(1) - low.shift(1) + 1e-10)
    
    # Sentiment features
    df['bull_power'] = high - df['ema_20']
    df['bear_power'] = low - df['ema_20']
    
    # Momentum divergence (price vs RSI)
    price_roc = close.pct_change(14)
    rsi_roc = df['rsi'].diff(14) / 100
    df['momentum_div'] = price_roc - rsi_roc
    
    # Volume Price Trend
    vpt = ((close - close.shift(1)) / close.shift(1) * volume).cumsum()
    df['vpt'] = vpt.diff(5) / (vpt.rolling(20).mean() + 1e-10)
    
    # Intraday momentum
    df['intraday_momentum'] = (close - df['open']) / (high - low + 1e-10)
    
    df.dropna(inplace=True)
    return df


# Feature columns for training (must match model expectations)
FEATURE_COLUMNS = [
    # Core (original)
    'rsi', 'adx', 'atr_pct', 'trend_str', 'vol_change',
    # v3.0 features
    'macd_hist_norm', 'bb_pct', 'bb_width',
    'roc_5', 'roc_10', 'obv_change',
    'price_position', 'body_pct',
    'above_ema200', 'ema_cross',
    # v3.1 features
    'ema20_slope', 'mfi', 'dist_50_high', 'dist_50_low',
    'hour_of_day', 'day_of_week',
    # v3.2 features
    'roc_21', 'roc_50', 'williams_r', 'cci', 'ultimate_osc',
    'volume_roc_5', 'volume_roc_21', 'chaikin_mf', 'force_index', 'ease_movement',
    'dist_sma20', 'dist_sma50', 'pivot_dist', 'fib_dist',
    'morning_volatility', 'afternoon_volatility', 'gap_up', 'gap_down', 'range_change',
    'bull_power', 'bear_power', 'momentum_div', 'vpt', 'intraday_momentum'
]
