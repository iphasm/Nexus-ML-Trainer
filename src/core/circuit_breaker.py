"""
Circuit Breaker pattern implementation for ML Cloud Trainer
Prevents cascading failures when external APIs are down
"""

import time
import asyncio
from enum import Enum
from typing import Any, Callable, Optional, Dict, Awaitable
from dataclasses import dataclass
from config.settings import get_config
from .logging_config import get_logger

logger = get_logger('circuit_breaker')


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: tuple = (Exception,)
    success_threshold: int = 3
    timeout: float = 30.0


class CircuitBreaker:
    """Circuit breaker implementation"""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.state != CircuitBreakerState.OPEN:
            return False

        if self.last_failure_time is None:
            return True

        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    def _record_success(self):
        """Record a successful call"""
        self.success_count += 1

        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self._reset()
        else:
            self.failure_count = 0

    def _record_failure(self):
        """Record a failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0

        if self.failure_count >= self.config.failure_threshold:
            self._trip()

    def _reset(self):
        """Reset circuit to closed state"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker {self.name} reset to CLOSED")

    def _trip(self):
        """Trip circuit to open state"""
        self.state = CircuitBreakerState.OPEN
        logger.warning(f"Circuit breaker {self.name} tripped to OPEN")

    def _can_attempt_call(self) -> bool:
        """Check if we can attempt a call"""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN")
                return True
            return False

        return True  # HALF_OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if not self._can_attempt_call():
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure()
            raise e

    async def call_async(self, func: Callable[..., Awaitable], *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        if not self._can_attempt_call():
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")

        try:
            # Apply timeout if configured
            if self.config.timeout > 0:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = await func(*args, **kwargs)

            self._record_success()
            return result
        except (asyncio.TimeoutError, *self.config.expected_exception) as e:
            self._record_failure()
            if isinstance(e, asyncio.TimeoutError):
                raise TimeoutError(f"Operation timed out after {self.config.timeout}s")
            raise e

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            }
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._load_default_breakers()

    def _load_default_breakers(self):
        """Load default circuit breakers for common services"""
        config = get_config()

        # Binance API circuit breaker
        self.breakers['binance'] = CircuitBreaker(
            'binance',
            CircuitBreakerConfig(
                failure_threshold=config.api.circuit_breaker_threshold,
                recovery_timeout=config.api.circuit_breaker_timeout,
                timeout=config.api.request_timeout
            )
        )

        # Bybit API circuit breaker
        self.breakers['bybit'] = CircuitBreaker(
            'bybit',
            CircuitBreakerConfig(
                failure_threshold=config.api.circuit_breaker_threshold,
                recovery_timeout=config.api.circuit_breaker_timeout,
                timeout=config.api.request_timeout
            )
        )

        # yFinance circuit breaker
        self.breakers['yfinance'] = CircuitBreaker(
            'yfinance',
            CircuitBreakerConfig(
                failure_threshold=3,  # Lower threshold for yfinance
                recovery_timeout=120,  # Longer recovery time
                timeout=15  # Shorter timeout
            )
        )

        logger.info("Circuit breaker registry initialized",
                   breaker_count=len(self.breakers))

    def get(self, name: str) -> CircuitBreaker:
        """Get circuit breaker by name"""
        if name not in self.breakers:
            # Create on-demand breaker with default config
            self.breakers[name] = CircuitBreaker(name)
            logger.info(f"Created new circuit breaker: {name}")

        return self.breakers[name]

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        return {
            name: breaker.get_status()
            for name, breaker in self.breakers.items()
        }


# Global registry instance
registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Get circuit breaker from global registry"""
    return registry.get(name)


def get_circuit_breaker_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers"""
    return registry.get_all_status()
