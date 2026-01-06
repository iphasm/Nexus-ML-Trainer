"""
Technical Indicators Module
Calculates all indicators required for ML model training.
Uses 'ta' library instead of 'pandas-ta' for better PyPI compatibility.
"""

import pandas as pd
import numpy as np
from ta.trend import ADXIndicator, EMAIndicator, SMAIndicator, MACD, CCIIndicator, DPOIndicator, KSTIndicator
from ta.momentum import RSIIndicator, WilliamsRIndicator, UltimateOscillator, StochRSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands, UlcerIndex
from ta.volume import MFIIndicator, ChaikinMoneyFlowIndicator, EaseOfMovementIndicator, ForceIndexIndicator, VolumeWeightedAveragePrice
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
import math


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX using ta library."""
    try:
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=period)
        return adx.adx().fillna(0).clip(0, 100)
    except Exception:
        return pd.Series(0, index=df.index)


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate MFI using ta library."""
    try:
        mfi = MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=period)
        return mfi.money_flow_index().fillna(50).clip(0, 100)
    except Exception:
        return pd.Series(50, index=df.index)


def calculate_stoch_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Stochastic RSI."""
    try:
        stoch_rsi = StochRSIIndicator(df['close'], window=period)
        return stoch_rsi.stochrsi_k().fillna(0.5)
    except Exception:
        return pd.Series(0.5, index=df.index)


def calculate_kst(df: pd.DataFrame) -> pd.Series:
    """Calculate Know Sure Thing (KST) oscillator."""
    try:
        kst = KSTIndicator(df['close'])
        return kst.kst().fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    try:
        cci = CCIIndicator(df['high'], df['low'], df['close'], window=period)
        return cci.cci().fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)


def calculate_dpo(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Detrended Price Oscillator."""
    try:
        dpo = DPOIndicator(df['close'], window=period)
        return dpo.dpo().fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)


def calculate_ulcer_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Ulcer Index for volatility measurement."""
    try:
        ulcer = UlcerIndex(df['close'], window=period)
        return ulcer.ulcer_index().fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)


def calculate_force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """Calculate Force Index."""
    try:
        force = ForceIndexIndicator(df['close'], df['volume'], window=period)
        return force.force_index().fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    try:
        vwap = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'])
        return vwap.volume_weighted_average_price().fillna(df['close'])
    except Exception:
        return df['close']


def calculate_market_regime(df: pd.DataFrame) -> pd.Series:
    """Calculate market regime based on volatility and trend."""
    try:
        # Combine ADX and ATR for regime classification
        adx = calculate_adx(df)
        atr_pct = (calculate_atr(df) / df['close']) * 100

        # Simple regime logic
        regime = pd.Series(0, index=df.index)  # Default: ranging

        # Trending up: ADX > 25 and positive slope
        trending_up = (adx > 25) & (df['close'] > df['close'].shift(20))
        regime[trending_up] = 1

        # Trending down: ADX > 25 and negative slope
        trending_down = (adx > 25) & (df['close'] < df['close'].shift(20))
        regime[trending_down] = -1

        # High volatility ranging
        high_vol = (atr_pct > atr_pct.rolling(50).mean() * 1.2)
        regime[high_vol] = 2

        return regime
    except Exception:
        return pd.Series(0, index=df.index)


def calculate_sentiment_proxy(df: pd.DataFrame) -> pd.Series:
    """Calculate sentiment proxy based on price action patterns."""
    try:
        # Simple sentiment proxy based on buying vs selling pressure
        returns = df['close'].pct_change()

        # Bullish signals
        bullish = (
            (df['close'] > df['open']) &  # Green candle
            (df['close'] > df['close'].shift(1)) &  # Higher close
            (df['volume'] > df['volume'].rolling(10).mean())  # Above average volume
        )

        # Bearish signals
        bearish = (
            (df['close'] < df['open']) &  # Red candle
            (df['close'] < df['close'].shift(1)) &  # Lower close
            (df['volume'] > df['volume'].rolling(10).mean())  # Above average volume
        )

        sentiment = pd.Series(0, index=df.index)
        sentiment[bullish] = 1
        sentiment[bearish] = -1

        # Smooth with rolling average
        return sentiment.rolling(5).mean().fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR using ta library."""
    try:
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=period)
        return atr.average_true_range().fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ALL technical indicators for training.
    EXTENDED FEATURE SET for v3.1
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # === BASIC INDICATORS ===
    # RSI (14)
    try:
        rsi_indicator = RSIIndicator(close, window=14)
        df['rsi'] = rsi_indicator.rsi().fillna(50)
    except Exception:
        df['rsi'] = 50
    
    # ATR (14)
    try:
        atr_indicator = AverageTrueRange(high, low, close, window=14)
        df['atr'] = atr_indicator.average_true_range().fillna(0)
    except Exception:
        df['atr'] = 0
    df['atr_pct'] = (df['atr'] / close) * 100
    
    # ADX
    df['adx'] = calculate_adx(df, period=14)
    
    # EMAs
    try:
        df['ema_9'] = EMAIndicator(close, window=9).ema_indicator()
        df['ema_20'] = EMAIndicator(close, window=20).ema_indicator()
        df['ema_50'] = EMAIndicator(close, window=50).ema_indicator()
        df['ema_200'] = EMAIndicator(close, window=200).ema_indicator()
    except Exception:
        df['ema_9'] = close
        df['ema_20'] = close
        df['ema_50'] = close
        df['ema_200'] = close
    
    # Fill NaN in EMAs
    for col in ['ema_9', 'ema_20', 'ema_50', 'ema_200']:
        df[col] = df[col].bfill().fillna(close)
    
    # Trend Strength (EMA divergence)
    df['trend_str'] = (df['ema_20'] - df['ema_50']) / close * 100
    
    # Volume Change
    df['vol_ma_5'] = volume.rolling(5).mean()
    df['vol_ma_20'] = volume.rolling(20).mean()
    df['vol_change'] = (df['vol_ma_5'] - df['vol_ma_20']) / (df['vol_ma_20'] + 1e-10)
    
    # === v3.0 FEATURES ===
    
    # MACD
    try:
        macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd().fillna(0)
        df['macd_signal'] = macd.macd_signal().fillna(0)
        df['macd_hist'] = macd.macd_diff().fillna(0)
    except Exception:
        df['macd'] = 0
        df['macd_signal'] = 0
        df['macd_hist'] = 0
    df['macd_hist_norm'] = df['macd_hist'] / close * 100
    
    # Bollinger Bands
    try:
        bb = BollingerBands(close, window=20, window_dev=2)
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
    except Exception:
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
    try:
        williams = WilliamsRIndicator(high, low, close, lbp=14)
        df['williams_r'] = williams.williams_r().fillna(-50)
    except Exception:
        df['williams_r'] = -50
    
    # CCI - calculated manually since ta library doesn't have direct CCI
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - tp_sma) / (0.015 * tp_mad + 1e-10)
    df['cci'] = df['cci'].fillna(0)
    
    # Ultimate Oscillator
    try:
        uo = UltimateOscillator(high, low, close)
        df['ultimate_osc'] = uo.ultimate_oscillator().fillna(50)
    except Exception:
        df['ultimate_osc'] = 50
    
    # Volume features
    df['volume_roc_5'] = (volume - volume.shift(5)) / (volume.shift(5) + 1e-10) * 100
    df['volume_roc_21'] = (volume - volume.shift(21)) / (volume.shift(21) + 1e-10) * 100
    
    # Chaikin Money Flow
    try:
        cmf = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20)
        df['chaikin_mf'] = cmf.chaikin_money_flow().fillna(0)
    except Exception:
        df['chaikin_mf'] = 0
    
    # Force Index
    df['force_index'] = close.diff() * volume
    df['force_index'] = df['force_index'].rolling(13).mean()
    
    # Ease of Movement
    try:
        eom = EaseOfMovementIndicator(high, low, volume, window=14)
        df['ease_movement'] = eom.ease_of_movement().fillna(0)
    except Exception:
        df['ease_movement'] = 0
    
    # Structure features
    try:
        sma_20 = SMAIndicator(close, window=20).sma_indicator()
        sma_50 = SMAIndicator(close, window=50).sma_indicator()
        df['dist_sma20'] = (close - sma_20) / close * 100
        df['dist_sma50'] = (close - sma_50) / close * 100
    except Exception:
        df['dist_sma20'] = 0
        df['dist_sma50'] = 0
    
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
    
    # === v3.3 MARKET REGIME FEATURE ===
    # Classifies market state as BULL (2), RANGE (1), or BEAR (0)
    # Based on EMA50 slope over 20 periods
    ema50_slope = (df['ema_50'] - df['ema_50'].shift(20)) / df['ema_50']
    df['market_regime'] = np.where(
        ema50_slope > 0.02, 2,  # BULL: EMA50 rising >2%
        np.where(ema50_slope < -0.02, 0, 1)  # BEAR: EMA50 falling >2%, else RANGE
    )

    # === ADVANCED FEATURES FOR v3.4 ===
    # Additional momentum indicators
    df['stoch_rsi'] = calculate_stoch_rsi(df, 14)
    df['kst'] = calculate_kst(df)
    df['cci'] = calculate_cci(df, 20)
    df['dpo'] = calculate_dpo(df, 20)

    # Advanced volatility
    df['ulcer_index'] = calculate_ulcer_index(df, 14)

    # Advanced volume indicators
    df['force_index'] = calculate_force_index(df, 13)
    df['vwap'] = calculate_vwap(df)

    # Market regime (enhanced version)
    df['market_regime_advanced'] = calculate_market_regime(df)
    df['sentiment_proxy'] = calculate_sentiment_proxy(df)

    # Feature interactions
    df['rsi_stoch_rsi'] = df['rsi'] * df['stoch_rsi']
    df['cci_kst'] = df['cci'] * df['kst']
    df['vol_price_change'] = df['atr_pct'] * df['roc_5'].abs()
    df['regime_volatility'] = df['market_regime'] * df['atr_pct']

    # Statistical features
    df['returns_skew'] = df['close'].pct_change().rolling(20).skew().fillna(0)
    df['returns_kurtosis'] = df['close'].pct_change().rolling(20).kurt().fillna(0)

    # Time-based features
    if hasattr(df.index, 'hour'):
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    else:
        df['hour_sin'] = 0
        df['hour_cos'] = 0
        df['day_sin'] = 0
        df['day_cos'] = 0

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
    'bull_power', 'bear_power', 'momentum_div', 'vpt', 'intraday_momentum',
    # v3.3 features
    'market_regime',
    # v3.4 ADVANCED FEATURES
    'stoch_rsi', 'kst', 'dpo',
    'ulcer_index', 'vwap',
    'market_regime_advanced', 'sentiment_proxy',
    'rsi_stoch_rsi', 'cci_kst', 'vol_price_change', 'regime_volatility',
    'returns_skew', 'returns_kurtosis',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]
