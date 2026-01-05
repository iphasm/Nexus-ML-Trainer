"""
Configuration settings for ML Cloud Trainer
Centralized configuration management with validation
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', ''))
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class APIConfig:
    """External API configuration"""
    binance_api_key: Optional[str] = field(default_factory=lambda: os.getenv('BINANCE_API_KEY'))
    binance_api_secret: Optional[str] = field(default_factory=lambda: os.getenv('BINANCE_API_SECRET'))
    request_timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 0.1
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60


@dataclass
class TrainingConfig:
    """ML training configuration"""
    max_candles: int = 50000
    symbols_limit: int = 25
    interval: str = "15m"
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    min_samples_per_class: int = 100
    feature_selection_threshold: float = 0.01


@dataclass
class ModelConfig:
    """Model hyperparameters and settings"""
    # XGBoost parameters
    max_depth: int = 6
    n_estimators: int = 200
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: int = 1
    gamma: float = 0.0

    # Training parameters
    early_stopping_rounds: int = 20
    eval_metric: str = 'mlogloss'


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    json_format: bool = True
    log_file: str = 'logs/ml_trainer.log'
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    enable_prometheus: bool = True
    metrics_port: int = 8001
    enable_health_checks: bool = True
    alert_on_failure: bool = True


@dataclass
class MLTrainerConfig:
    """Main configuration class"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / 'logs')
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / 'models')

    # Default symbols for training
    default_symbols: List[str] = field(default_factory=lambda: [
        # Major caps
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',

        # Meme coins
        'DOGEUSDT', 'WIFUSDT', '1000PEPEUSDT', 'PONKEUSDT',

        # DeFi
        'AAVEUSDT', 'CRVUSDT', 'SNXUSDT', 'LDOUSDT', 'DYDXUSDT',

        # AI/Tech
        'WLDUSDT', 'INJUSDT',

        # L1/Infrastructure
        'NEARUSDT', 'ARBUSDT', 'LTCUSDT', 'BCHUSDT', 'ETCUSDT', 'ALGOUSDT',

        # Bybit exclusive
        'FLOWUSDT', 'PENDLEUSDT', 'XAIUSDT'
    ])

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters"""
        # Database validation
        if not self.database.url:
            raise ValueError("DATABASE_URL environment variable is required")

        # Training validation
        if self.training.max_candles < 1000 or self.training.max_candles > 100000:
            raise ValueError("max_candles must be between 1000 and 100000")

        if self.training.symbols_limit < 1 or self.training.symbols_limit > 50:
            raise ValueError("symbols_limit must be between 1 and 50")

        if self.training.test_size <= 0 or self.training.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")

        # Model validation
        if self.model.max_depth < 1 or self.model.max_depth > 15:
            raise ValueError("max_depth must be between 1 and 15")

        if self.model.n_estimators < 50 or self.model.n_estimators > 1000:
            raise ValueError("n_estimators must be between 50 and 1000")

        if self.model.learning_rate <= 0 or self.model.learning_rate > 1:
            raise ValueError("learning_rate must be between 0 and 1")

        # Create directories
        self.logs_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)


# Global configuration instance
config = MLTrainerConfig()


def get_config() -> MLTrainerConfig:
    """Get the global configuration instance"""
    return config


def reload_config() -> MLTrainerConfig:
    """Reload configuration from environment"""
    global config
    config = MLTrainerConfig()
    return config
