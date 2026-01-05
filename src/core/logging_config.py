"""
Structured logging configuration for ML Cloud Trainer
Provides JSON-formatted logs with proper rotation and structured data
"""

import logging
import json
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional
from config.settings import get_config


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        """Format log record as JSON"""
        # Base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)


class StructuredLogger:
    """Structured logger with context support"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}

    def with_context(self, **context) -> 'StructuredLogger':
        """Return logger with additional context"""
        new_logger = StructuredLogger(self.logger.name)
        new_logger.context.update(self.context)
        new_logger.context.update(context)
        return new_logger

    def _log_with_context(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log with context information"""
        log_extra = {'extra_fields': dict(self.context)}
        if extra:
            log_extra['extra_fields'].update(extra)
        if kwargs:
            log_extra['extra_fields'].update(kwargs)

        self.logger.log(level, message, extra=log_extra)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, exc: Optional[Exception] = None, **kwargs):
        """Log error message"""
        if exc:
            kwargs['exception_type'] = type(exc).__name__
            kwargs['exception_message'] = str(exc)
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics"""
        self.info(f"Performance: {operation}",
                 operation=operation,
                 duration=duration,
                 **metrics)

    def log_training_progress(self, epoch: int, loss: float, accuracy: float, **metrics):
        """Log training progress"""
        self.info("Training progress",
                 epoch=epoch,
                 loss=loss,
                 accuracy=accuracy,
                 **metrics)

    def log_api_call(self, api_name: str, method: str, status_code: Optional[int] = None,
                    duration: Optional[float] = None, **details):
        """Log API call details"""
        self.info(f"API call: {api_name}.{method}",
                 api_name=api_name,
                 method=method,
                 status_code=status_code,
                 duration=duration,
                 **details)


def setup_logging(config=None) -> StructuredLogger:
    """Setup comprehensive logging configuration"""
    if config is None:
        config = get_config()

    # Create logs directory
    config.logs_dir.mkdir(exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.logging.level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler (human-readable)
    console_formatter = logging.Formatter(config.logging.format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # File handler (JSON structured)
    if config.logging.json_format:
        file_formatter = JSONFormatter()
    else:
        file_formatter = logging.Formatter(config.logging.format)

    log_file_path = config.logs_dir / config.logging.log_file
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Create structured logger instance
    logger = StructuredLogger('ml_trainer')

    # Log startup
    logger.info("Logging system initialized",
               log_level=config.logging.level,
               log_file=str(log_file_path),
               json_format=config.logging.json_format)

    return logger


# Global logger instance
logger = setup_logging()


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)


def log_api_error(api_name: str, error: Exception, **context):
    """Log API errors with context"""
    logger.error(f"API error in {api_name}",
                api_name=api_name,
                exception_type=type(error).__name__,
                exception_message=str(error),
                **context)


def log_training_error(error: Exception, **context):
    """Log training errors with context"""
    logger.error("Training error occurred",
                exception_type=type(error).__name__,
                exception_message=str(error),
                **context)


def log_performance_metric(metric_name: str, value: float, **context):
    """Log performance metrics"""
    logger.info(f"Performance metric: {metric_name} = {value}",
               metric_name=metric_name,
               value=value,
               **context)
