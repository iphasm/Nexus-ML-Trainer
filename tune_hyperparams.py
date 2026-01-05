"""
Hyperparameter Optimization Script using Optuna
Runs separately to find optimal XGBoost parameters for Nexus Cortex.
"""

import os
import sys
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

from indicators import add_indicators, FEATURE_COLUMNS
from train_cortex import fetch_crypto_data, label_data, DEFAULT_SYMBOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_data(symbols: list, max_candles: int = 10000):
    """Prepare training data using same pipeline as train_cortex."""
    all_data = []
    
    print(f"📥 Loading data for {len(symbols)} symbols...")
    for symbol in symbols[:10]:  # Use subset for faster tuning
        try:
            df = fetch_crypto_data(symbol, max_candles=max_candles)
            if df is not None and len(df) > 100:
                df = add_indicators(df)
                df = label_data(df)
                if len(df) > 100:
                    all_data.append(df)
                    print(f"  ✅ {symbol}: {len(df)} samples")
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")
    
    if not all_data:
        raise ValueError("No data collected!")
    
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Filter rare classes
    min_samples_pct = 0.01
    min_samples = int(len(full_df) * min_samples_pct)
    class_counts = full_df['target'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index.tolist()
    full_df = full_df[full_df['target'].isin(valid_classes)]
    
    X = full_df[FEATURE_COLUMNS]
    y = full_df['target']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    sample_weights = compute_sample_weight('balanced', y_encoded)
    
    print(f"📊 Total samples: {len(full_df):,}")
    print(f"🎯 Classes: {list(label_encoder.classes_)}")
    
    return X_scaled, y_encoded, sample_weights, label_encoder


def objective(trial, X, y, sample_weights, n_classes):
    """Optuna objective function for XGBoost hyperparameter optimization."""
    
    params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }
    
    model = XGBClassifier(**params)
    
    # TimeSeriesSplit cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        weights = sample_weights[train_idx]
        
        model.fit(X_train, y_train, sample_weight=weights)
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return np.mean(scores)


def run_optimization(n_trials: int = 50):
    """Run Optuna hyperparameter optimization."""
    
    print("=" * 70)
    print("🔧 NEXUS CORTEX HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"⏰ Started at: {datetime.now()}")
    print(f"🔢 Trials: {n_trials}")
    print()
    
    # Prepare data
    X, y, weights, label_encoder = prepare_data(DEFAULT_SYMBOLS, max_candles=10000)
    n_classes = len(label_encoder.classes_)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='nexus_xgboost_tuning',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    print(f"\n🚀 Starting {n_trials} optimization trials...")
    study.optimize(
        lambda trial: objective(trial, X, y, weights, n_classes),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "=" * 70)
    print("📊 OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"🏆 Best accuracy: {study.best_value:.4f}")
    print(f"\n🔧 Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Generate Python code for best params
    print(f"\n📝 Copy this to train_cortex.py:")
    print("-" * 50)
    print("model = XGBClassifier(")
    print("    objective='multi:softprob',")
    print(f"    num_class={n_classes},")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"    {key}={value:.6f},")
        else:
            print(f"    {key}={value},")
    print("    random_state=42,")
    print("    n_jobs=-1,")
    print("    use_label_encoder=False,")
    print("    eval_metric='mlogloss'")
    print(")")
    print("-" * 50)
    
    return study.best_params, study.best_value


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    args = parser.parse_args()
    
    try:
        best_params, best_score = run_optimization(n_trials=args.trials)
        print(f"\n✅ Optimization complete! Best CV score: {best_score:.4f}")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
