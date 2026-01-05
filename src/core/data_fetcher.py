"""
Enhanced data fetching module with circuit breaker and robust error handling
Supports multiple data sources with fallback mechanisms
"""

import asyncio
import time
import pandas as pd
import yfinance as yf
from binance.client import Client
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from config.settings import get_config
from .logging_config import get_logger
from .exceptions import DataFetchError
from .circuit_breaker import get_circuit_breaker
from .validators import validate_symbol

logger = get_logger('data_fetcher')
config = get_config()


class DataFetcher:
    """Enhanced data fetcher with circuit breaker protection"""

    def __init__(self):
        self.binance_breaker = get_circuit_breaker('binance')
        self.yfinance_breaker = get_circuit_breaker('yfinance')
        self.binance_client = self._create_binance_client()

    def _create_binance_client(self) -> Optional[Client]:
        """Create Binance client if credentials are available"""
        if config.api.binance_api_key and config.api.binance_api_secret:
            return Client(
                api_key=config.api.binance_api_key,
                api_secret=config.api.binance_api_secret,
                requests_params={'timeout': config.api.request_timeout}
            )
        logger.warning("Binance API credentials not found, using public access only")
        return None

    async def fetch_crypto_data(self, symbol: str, max_candles: int = 15000,
                               interval: str = "15m") -> Optional[pd.DataFrame]:
        """Fetch crypto data with fallback sources and circuit breaker protection"""
        start_time = time.time()

        try:
            # Validate input
            symbol = validate_symbol(symbol)

            logger.info(f"Fetching data for {symbol}",
                       symbol=symbol, max_candles=max_candles, interval=interval)

            # Try Binance first
            df = await self._fetch_from_binance(symbol, max_candles, interval)

            if df is not None and len(df) >= 100:
                duration = time.time() - start_time
                logger.log_performance("data_fetch", duration,
                                     symbol=symbol, source="binance", candles=len(df))
                return df

            # Fallback to yFinance
            logger.warning(f"Binance failed for {symbol}, trying yFinance")
            df = await self._fetch_from_yfinance(symbol, max_candles, interval)

            if df is not None and len(df) >= 100:
                duration = time.time() - start_time
                logger.log_performance("data_fetch", duration,
                                     symbol=symbol, source="yfinance", candles=len(df))
                return df

            logger.error(f"All data sources failed for {symbol}")
            return None

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Data fetch failed for {symbol}",
                        symbol=symbol, duration=duration, error=str(e))
            return None

    async def _fetch_from_binance(self, symbol: str, max_candles: int,
                                 interval: str) -> Optional[pd.DataFrame]:
        """Fetch data from Binance with circuit breaker"""

        def _binance_call():
            if not self.binance_client:
                raise DataFetchError("Binance client not available")

            # Calculate start time
            interval_minutes = self._interval_to_minutes(interval)
            total_minutes = max_candles * interval_minutes
            start_time = datetime.utcnow() - timedelta(minutes=total_minutes)
            start_str = start_time.strftime('%d %b %Y %H:%M:%S')

            # Fetch klines
            klines = self.binance_client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                limit=max_candles
            )

            if not klines:
                raise DataFetchError(f"No data received from Binance for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Clean and format data
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna().sort_values('timestamp').reset_index(drop=True)

            return df

        try:
            return await self.binance_breaker.call_async(
                lambda: asyncio.get_event_loop().run_in_executor(None, _binance_call)
            )
        except Exception as e:
            logger.warning(f"Binance fetch failed for {symbol}: {e}")
            raise DataFetchError(f"Binance fetch failed: {e}")

    async def _fetch_from_yfinance(self, symbol: str, max_candles: int,
                                  interval: str) -> Optional[pd.DataFrame]:
        """Fetch data from yFinance with circuit breaker and smart symbol mapping"""

        def _yfinance_call():
            # Convert symbol to yFinance format
            yf_symbol = self._convert_to_yfinance_symbol(symbol)

            # Map interval to yFinance period
            period = self._get_yfinance_period(max_candles, interval)
            yf_interval = self._convert_interval_to_yfinance(interval)

            logger.debug(f"Fetching {yf_symbol} from yFinance",
                        symbol=symbol, yf_symbol=yf_symbol, period=period, interval=yf_interval)

            ticker = yf.Ticker(yf_symbol)

            # Try different periods if the first one fails
            for attempt_period in [period, "60d", "30d", "7d"]:
                try:
                    df = ticker.history(period=attempt_period, interval=yf_interval)

                    if not df.empty and len(df) > 10:
                        break
                except Exception:
                    continue
            else:
                raise DataFetchError(f"No data from yFinance for {yf_symbol}")

            # Convert to our format
            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]

            # Rename columns
            rename_map = {
                'datetime': 'timestamp',
                'date': 'timestamp'
            }
            df = df.rename(columns=rename_map)

            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise DataFetchError(f"Missing required columns in yFinance data for {symbol}")

            df = df[required_cols].copy()

            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna().sort_values('timestamp').reset_index(drop=True)

            # Limit to max_candles
            if len(df) > max_candles:
                df = df.tail(max_candles).reset_index(drop=True)

            return df

        try:
            return await self.yfinance_breaker.call_async(
                lambda: asyncio.get_event_loop().run_in_executor(None, _yfinance_call)
            )
        except Exception as e:
            logger.warning(f"yFinance fetch failed for {symbol}: {e}")
            raise DataFetchError(f"yFinance fetch failed: {e}")

    def _convert_to_yfinance_symbol(self, symbol: str) -> str:
        """Convert trading symbol to yFinance format"""
        conversions = {
            'BTCUSDT': 'BTC-USD', 'ETHUSDT': 'ETH-USD', 'BNBUSDT': 'BNB-USD',
            'ADAUSDT': 'ADA-USD', 'SOLUSDT': 'SOL-USD', 'DOTUSDT': 'DOT-USD',
            'DOGEUSDT': 'DOGE-USD', 'AVAXUSDT': 'AVAX-USD', 'LINKUSDT': 'LINK-USD',
            'LTCUSDT': 'LTC-USD', 'BCHUSDT': 'BCH-USD', 'ETCUSDT': 'ETC-USD',
            'XRPUSDT': 'XRP-USD'
        }

        if symbol in conversions:
            return conversions[symbol]

        if symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}-USD"

        return symbol

    def _get_yfinance_period(self, max_candles: int, interval: str) -> str:
        """Get appropriate yFinance period based on candles and interval"""
        interval_minutes = self._interval_to_minutes(interval)
        total_minutes = max_candles * interval_minutes

        if total_minutes <= 60 * 24:
            return "1d"
        elif total_minutes <= 60 * 24 * 7:
            return "7d"
        elif total_minutes <= 60 * 24 * 30:
            return "30d"
        elif total_minutes <= 60 * 24 * 60:
            return "60d"
        else:
            return "max"

    def _convert_interval_to_yfinance(self, interval: str) -> str:
        """Convert interval to yFinance format"""
        conversions = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '1d': '1d'
        }
        return conversions.get(interval, '1h')

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes"""
        conversions = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return conversions.get(interval, 15)

    async def fetch_multiple_symbols(self, symbols: List[str], max_candles: int = 15000,
                                   interval: str = "15m", max_concurrent: int = 5) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols concurrently"""
        logger.info(f"Fetching data for {len(symbols)} symbols concurrently",
                   symbol_count=len(symbols), max_concurrent=max_concurrent)

        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}

        async def fetch_with_semaphore(symbol):
            async with semaphore:
                df = await self.fetch_crypto_data(symbol, max_candles, interval)
                return symbol, df

        tasks = [fetch_with_semaphore(symbol) for symbol in symbols]
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = 0
        failed = 0

        for result in completed_results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                failed += 1
                continue

            symbol, df = result
            if df is not None:
                results[symbol] = df
                successful += 1
            else:
                failed += 1

        logger.info("Multi-symbol fetch completed",
                   successful=successful, failed=failed, total=len(symbols))

        return results


# Global data fetcher instance
data_fetcher = DataFetcher()
