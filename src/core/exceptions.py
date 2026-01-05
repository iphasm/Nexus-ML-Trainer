"""
Custom exceptions for ML Cloud Trainer
Provides specific exception types for better error handling and debugging
"""

from typing import Optional, Dict, Any


class MLTrainerError(Exception):
    """Base exception for all ML Trainer errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details
        }


class ConfigurationError(MLTrainerError):
    """Raised when configuration is invalid or missing"""
    pass


class DataFetchError(MLTrainerError):
    """Raised when data fetching fails"""
    pass


class DataValidationError(MLTrainerError):
    """Raised when data validation fails"""
    pass


class ModelTrainingError(MLTrainerError):
    """Raised when model training fails"""
    pass


class ModelValidationError(MLTrainerError):
    """Raised when model validation fails"""
    pass


class DatabaseError(MLTrainerError):
    """Raised when database operations fail"""
    pass


class CircuitBreakerError(MLTrainerError):
    """Raised when circuit breaker is open"""
    pass


class TimeoutError(MLTrainerError):
    """Raised when operations timeout"""
    pass


class ResourceExhaustionError(MLTrainerError):
    """Raised when system resources are exhausted"""
    pass


class ValidationError(MLTrainerError):
    """Raised when input validation fails"""
    pass


# Convenience functions for creating exceptions
def config_error(message: str, **details) -> ConfigurationError:
    """Create a configuration error"""
    return ConfigurationError(message, details)


def data_fetch_error(message: str, **details) -> DataFetchError:
    """Create a data fetch error"""
    return DataFetchError(message, details)


def validation_error(message: str, **details) -> ValidationError:
    """Create a validation error"""
    return ValidationError(message, details)


def database_error(message: str, **details) -> DatabaseError:
    """Create a database error"""
    return DatabaseError(message, details)


def training_error(message: str, **details) -> ModelTrainingError:
    """Create a training error"""
    return ModelTrainingError(message, details)
