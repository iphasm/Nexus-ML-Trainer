"""
Basic tests for core modules - Critical fixes validation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from config.settings import MLTrainerConfig
from src.core.exceptions import ValidationError, DataFetchError
from src.core.validators import InputValidator, validate_training_request
from src.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from src.core.logging_config import StructuredLogger


class TestInputValidation:
    """Test critical input validation"""

    def setup_method(self):
        self.config = MLTrainerConfig()
        self.validator = InputValidator(self.config)

    def test_validate_training_config_valid(self):
        """Test valid training configuration"""
        config = {
            'candles': 10000,
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'interval': '15m'
        }

        result = self.validator.validate_training_config(config)

        assert result['candles'] == 10000
        assert result['symbols'] == ['BTCUSDT', 'ETHUSDT']
        assert result['interval'] == '15m'

    def test_validate_training_config_invalid_candles(self):
        """Test invalid candles validation"""
        config = {'candles': 50000}  # Exceeds max

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_training_config(config)

        assert "candles must be integer between 1000" in str(exc_info.value)

    def test_validate_training_config_invalid_symbols(self):
        """Test invalid symbols validation"""
        config = {'symbols': ['INVALID']}  # Invalid format

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_training_config(config)

        assert "symbol must be valid trading pair" in str(exc_info.value)

    def test_validate_symbol_valid(self):
        """Test valid symbol validation"""
        assert self.validator._validate_symbol('BTCUSDT') == 'BTCUSDT'
        assert self.validator._validate_symbol('ethusdt') == 'ETHUSDT'  # Case insensitive

    def test_validate_symbol_invalid(self):
        """Test invalid symbol validation"""
        with pytest.raises(ValidationError):
            self.validator._validate_symbol('INVALID')

        with pytest.raises(ValidationError):
            self.validator._validate_symbol('BTCUSDTTOOLONG')


class TestCircuitBreaker:
    """Test circuit breaker functionality"""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state"""
        cb = CircuitBreaker('test')

        assert cb.state.value == 'closed'
        assert cb._can_attempt_call() == True

    def test_circuit_breaker_open_state(self):
        """Test circuit breaker opens after failures"""
        cb = CircuitBreaker('test', Mock(failure_threshold=2))

        # First failure
        cb._record_failure()
        assert cb.state.value == 'closed'

        # Second failure - should open
        cb._record_failure()
        assert cb.state.value == 'open'
        assert cb._can_attempt_call() == False

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery to half-open"""
        cb = CircuitBreaker('test', Mock(failure_threshold=1, recovery_timeout=0))

        cb._record_failure()
        assert cb.state.value == 'open'

        # Should allow attempt after timeout
        assert cb._can_attempt_call() == True
        assert cb.state.value == 'half_open'

    def test_circuit_breaker_success_recovery(self):
        """Test circuit breaker recovers after success"""
        cb = CircuitBreaker('test', Mock(failure_threshold=1, success_threshold=2))

        cb._record_failure()
        cb._record_success()  # Half-open success
        assert cb.state.value == 'half_open'

        cb._record_success()  # Second success - should close
        assert cb.state.value == 'closed'

    @pytest.mark.asyncio
    async def test_circuit_breaker_call_success(self):
        """Test successful circuit breaker call"""
        cb = CircuitBreaker('test')

        def success_func():
            return "success"

        result = cb.call(success_func)
        assert result == "success"
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_call_failure(self):
        """Test failed circuit breaker call"""
        cb = CircuitBreaker('test', Mock(failure_threshold=1))

        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.failure_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_blocks_calls(self):
        """Test that open circuit breaker blocks calls"""
        cb = CircuitBreaker('test', Mock(failure_threshold=1))

        cb._record_failure()  # Open the circuit

        def func():
            return "should not execute"

        with pytest.raises(CircuitBreakerOpenError):
            cb.call(func)


class TestStructuredLogging:
    """Test structured logging functionality"""

    def setup_method(self):
        self.logger = StructuredLogger('test')

    def test_logger_creation(self):
        """Test logger creation"""
        assert self.logger.logger.name == 'test'

    @patch('src.core.logging_config.logger')
    def test_log_with_context(self, mock_logger):
        """Test logging with context"""
        self.logger = StructuredLogger('test')
        self.logger.context = {'user_id': 123}

        self.logger.info("Test message", action="login", ip="192.168.1.1")

        # Verify the call includes context
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == 20  # INFO level
        assert "Test message" in call_args[0][1]

        # Check extra fields include context
        extra = call_args[1]['extra']
        assert extra['extra_fields']['user_id'] == 123
        assert extra['extra_fields']['action'] == "login"
        assert extra['extra_fields']['ip'] == "192.168.1.1"

    @patch('src.core.logging_config.logger')
    def test_log_performance(self, mock_logger):
        """Test performance logging"""
        self.logger.log_performance("data_fetch", 1.5, records=1000)

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        extra = call_args[1]['extra']['extra_fields']

        assert extra['operation'] == "data_fetch"
        assert extra['duration'] == 1.5
        assert extra['records'] == 1000

    @patch('src.core.logging_config.logger')
    def test_log_api_error(self, mock_logger):
        """Test API error logging"""
        error = ValueError("API timeout")

        self.logger.error("API call failed",
                         api_name="binance",
                         exception_type=type(error).__name__,
                         exception_message=str(error),
                         symbol="BTCUSDT")

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        extra = call_args[1]['extra']['extra_fields']

        assert extra['api_name'] == "binance"
        assert extra['exception_type'] == "ValueError"
        assert extra['exception_message'] == "API timeout"
        assert extra['symbol'] == "BTCUSDT"


class TestDataFetcher:
    """Test data fetcher functionality"""

    def setup_method(self):
        self.fetcher = DataFetcher()

    def test_symbol_conversion(self):
        """Test yFinance symbol conversion"""
        assert self.fetcher._convert_to_yfinance_symbol('BTCUSDT') == 'BTC-USD'
        assert self.fetcher._convert_to_yfinance_symbol('ETHUSDT') == 'ETH-USD'
        assert self.fetcher._convert_to_yfinance_symbol('UNKNOWNUSDT') == 'UNKNOWN-USD'

    def test_interval_conversion(self):
        """Test interval to minutes conversion"""
        assert self.fetcher._interval_to_minutes('1m') == 1
        assert self.fetcher._interval_to_minutes('1h') == 60
        assert self.fetcher._interval_to_minutes('1d') == 1440
        assert self.fetcher._interval_to_minutes('unknown') == 15  # default

    @pytest.mark.asyncio
    @patch('src.core.data_fetcher.Client')
    async def test_binance_fetch_failure_fallback(self, mock_client_class):
        """Test fallback to yFinance when Binance fails"""
        # Mock Binance failure
        mock_client = Mock()
        mock_client.get_historical_klines.side_effect = Exception("Binance error")
        mock_client_class.return_value = mock_client

        # Mock yFinance success
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='15min'),
            'open': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        })

        with patch.object(self.fetcher, '_fetch_from_yfinance', new_callable=AsyncMock) as mock_yf:
            mock_yf.return_value = sample_data

            result = await self.fetcher.fetch_crypto_data('BTCUSDT', 100)

            assert result is not None
            assert len(result) == 100
            mock_yf.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_symbol_validation(self):
        """Test symbol validation in fetch"""
        with pytest.raises(Exception):  # Should be caught and logged
            await self.fetcher.fetch_crypto_data('INVALID', 100)


class TestFeatureEngineering:
    """Test feature engineering functionality"""

    def test_add_indicators_basic(self):
        """Test basic indicator calculation"""
        from src.core.feature_engineering import add_indicators

        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='15min')
        data = {
            'timestamp': dates,
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(85, 95, 100),
            'close': np.random.uniform(95, 105, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }
        df = pd.DataFrame(data)

        result = add_indicators(df)

        # Check that key features are present
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert 'bb_upper' in result.columns
        assert 'adx' in result.columns

        # Check that no NaN values remain (should be filled)
        assert not result.isnull().any().any()

    def test_validate_features_success(self):
        """Test feature validation success"""
        from src.core.feature_engineering import validate_features, add_indicators, FEATURE_COLUMNS

        # Create sample data with enough rows for indicators
        dates = pd.date_range('2023-01-01', periods=200, freq='15min')
        data = {
            'timestamp': dates,
            'open': np.random.uniform(90, 110, 200),
            'high': np.random.uniform(105, 115, 200),
            'low': np.random.uniform(85, 95, 200),
            'close': np.random.uniform(95, 105, 200),
            'volume': np.random.uniform(1000, 5000, 200)
        }
        df = pd.DataFrame(data)
        df = add_indicators(df)

        # Should validate successfully
        assert validate_features(df) == True

    def test_get_feature_importance_ranking(self):
        """Test feature importance ranking"""
        from src.core.feature_engineering import get_feature_importance_ranking

        ranking = get_feature_importance_ranking()

        assert isinstance(ranking, list)
        assert len(ranking) > 0
        assert 'rsi' in ranking  # Should contain key indicators


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
