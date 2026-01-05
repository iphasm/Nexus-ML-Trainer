"""
Input validation utilities for ML Cloud Trainer
Ensures all inputs are safe and within acceptable ranges
"""

import re
from typing import List, Dict, Any, Union, Optional
from config.settings import MLTrainerConfig
from .exceptions import ValidationError


class InputValidator:
    """Centralized input validation"""

    def __init__(self, config: MLTrainerConfig):
        self.config = config

    def validate_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training configuration parameters"""
        validated = {}

        # Validate candles
        candles = config.get('candles', self.config.training.max_candles)
        if not isinstance(candles, int) or candles < 1000 or candles > self.config.training.max_candles:
            raise ValidationError(
                f"candles must be integer between 1000 and {self.config.training.max_candles}",
                field='candles', value=candles
            )
        validated['candles'] = candles

        # Validate symbols
        symbols = config.get('symbols', self.config.default_symbols[:self.config.training.symbols_limit])
        if not isinstance(symbols, list):
            raise ValidationError("symbols must be a list", field='symbols', value=type(symbols))

        if len(symbols) == 0:
            raise ValidationError("symbols list cannot be empty", field='symbols')

        if len(symbols) > self.config.training.symbols_limit:
            raise ValidationError(
                f"symbols list cannot exceed {self.config.training.symbols_limit} items",
                field='symbols', count=len(symbols)
            )

        # Validate each symbol format
        for symbol in symbols:
            self._validate_symbol(symbol)

        validated['symbols'] = symbols

        # Validate interval
        interval = config.get('interval', self.config.training.interval)
        if not self._is_valid_interval(interval):
            raise ValidationError(
                "interval must be a valid timeframe (e.g., '1m', '5m', '15m', '1h', '1d')",
                field='interval', value=interval
            )
        validated['interval'] = interval

        # Validate test_size
        test_size = config.get('test_size', self.config.training.test_size)
        if not isinstance(test_size, (int, float)) or test_size <= 0 or test_size >= 1:
            raise ValidationError(
                "test_size must be float between 0 and 1",
                field='test_size', value=test_size
            )
        validated['test_size'] = float(test_size)

        return validated

    def validate_symbol_list(self, symbols: List[str]) -> List[str]:
        """Validate a list of trading symbols"""
        if not isinstance(symbols, list):
            raise ValidationError("symbols must be a list", value=type(symbols))

        if len(symbols) == 0:
            raise ValidationError("symbols list cannot be empty")

        validated_symbols = []
        for symbol in symbols:
            validated_symbols.append(self._validate_symbol(symbol))

        return validated_symbols

    def validate_candles_count(self, candles: int) -> int:
        """Validate candles count parameter"""
        if not isinstance(candles, int):
            raise ValidationError("candles must be integer", value=type(candles))

        if candles < 100 or candles > self.config.training.max_candles:
            raise ValidationError(
                f"candles must be between 100 and {self.config.training.max_candles}",
                value=candles
            )

        return candles

    def validate_percentage(self, value: Union[int, float], field_name: str) -> float:
        """Validate percentage values (0-100)"""
        try:
            percentage = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{field_name} must be numeric", field=field_name, value=value)

        if percentage < 0 or percentage > 100:
            raise ValidationError(
                f"{field_name} must be between 0 and 100",
                field=field_name, value=percentage
            )

        return percentage

    def validate_positive_number(self, value: Union[int, float], field_name: str) -> float:
        """Validate positive numeric values"""
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{field_name} must be numeric", field=field_name, value=value)

        if number <= 0:
            raise ValidationError(
                f"{field_name} must be positive",
                field=field_name, value=number
            )

        return number

    def validate_leverage(self, leverage: Union[int, float]) -> float:
        """Validate leverage values"""
        try:
            lev = float(leverage)
        except (TypeError, ValueError):
            raise ValidationError("leverage must be numeric", value=leverage)

        if lev < 1 or lev > 125:
            raise ValidationError("leverage must be between 1 and 125", value=lev)

        return lev

    def sanitize_string(self, text: str, max_length: int = 255) -> str:
        """Sanitize string input to prevent injection attacks"""
        if not isinstance(text, str):
            raise ValidationError("input must be string", value=type(text))

        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>]', '', text)

        if len(sanitized) > max_length:
            raise ValidationError(
                f"string too long (max {max_length} characters)",
                length=len(sanitized)
            )

        return sanitized.strip()

    def _validate_symbol(self, symbol: str) -> str:
        """Validate individual trading symbol format"""
        if not isinstance(symbol, str):
            raise ValidationError("symbol must be string", value=type(symbol))

        # Binance/Bybit symbol pattern: BASEQUOTE (e.g., BTCUSDT)
        symbol = symbol.upper().strip()

        if not re.match(r'^[A-Z]{2,10}(USDT|BUSD|BTC|ETH|BNB)$', symbol):
            raise ValidationError(
                "symbol must be valid trading pair (e.g., BTCUSDT, ETHUSDT)",
                symbol=symbol
            )

        if len(symbol) > 15:
            raise ValidationError("symbol too long", symbol=symbol)

        return symbol

    def _is_valid_interval(self, interval: str) -> bool:
        """Check if interval is a valid timeframe"""
        valid_intervals = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]
        return interval in valid_intervals


# Global validator instance
_config = MLTrainerConfig()
validator = InputValidator(_config)


def validate_training_request(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to validate training requests"""
    return validator.validate_training_config(config)


def validate_symbol(symbol: str) -> str:
    """Convenience function to validate trading symbols"""
    return validator._validate_symbol(symbol)


def validate_candles(candles: int) -> int:
    """Convenience function to validate candles count"""
    return validator.validate_candles_count(candles)
