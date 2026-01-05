"""
Model Uploader - Saves trained ML models to PostgreSQL
Allows Nexus-TB to download and use the latest model without file sharing.
"""

import os
import io
import json
import joblib
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get PostgreSQL connection from DATABASE_URL environment variable."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    # Railway uses postgres:// but psycopg2 needs postgresql://
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    return psycopg2.connect(database_url)


def init_ml_models_table():
    """Create the ml_models table if it doesn't exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(50) NOT NULL,
                    model_blob BYTEA NOT NULL,
                    scaler_blob BYTEA NOT NULL,
                    accuracy FLOAT,
                    cv_score FLOAT,
                    feature_names TEXT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    is_active BOOLEAN DEFAULT TRUE
                );
                
                CREATE INDEX IF NOT EXISTS idx_ml_models_version ON ml_models(version);
                CREATE INDEX IF NOT EXISTS idx_ml_models_created_at ON ml_models(created_at DESC);
            """)
            conn.commit()
            logger.info("✅ ml_models table initialized")
    except Exception as e:
        logger.error(f"❌ Error initializing ml_models table: {e}")
        raise
    finally:
        conn.close()


def upload_model(
    model_data: Dict[str, Any],
    scaler: Any,
    version: str,
    accuracy: float = None,
    cv_score: float = None,
    feature_names: list = None,
    metadata: Dict[str, Any] = None
) -> bool:
    """
    Upload a trained model to PostgreSQL.
    
    Args:
        model_data: Dictionary containing 'model', 'label_encoder', 'feature_names'
        scaler: The fitted scaler (RobustScaler)
        version: Version string (e.g., "v3.1-20260104")
        accuracy: Test accuracy score
        cv_score: Cross-validation score
        feature_names: List of feature names used
        metadata: Additional metadata (training params, etc.)
    
    Returns:
        True if upload successful, False otherwise
    """
    conn = get_db_connection()
    try:
        # Serialize model and scaler to bytes
        model_buffer = io.BytesIO()
        joblib.dump(model_data, model_buffer)
        model_blob = model_buffer.getvalue()
        
        scaler_buffer = io.BytesIO()
        joblib.dump(scaler, scaler_buffer)
        scaler_blob = scaler_buffer.getvalue()
        
        # Convert numpy types to Python native types for PostgreSQL compatibility
        accuracy_float = float(accuracy) if accuracy is not None else None
        cv_score_float = float(cv_score) if cv_score is not None else None
        
        # Deactivate previous models
        with conn.cursor() as cur:
            cur.execute("UPDATE ml_models SET is_active = FALSE WHERE is_active = TRUE")
            
            # Insert new model
            cur.execute("""
                INSERT INTO ml_models 
                (version, model_blob, scaler_blob, accuracy, cv_score, feature_names, metadata, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE)
                RETURNING id
            """, (
                version,
                psycopg2.Binary(model_blob),
                psycopg2.Binary(scaler_blob),
                accuracy_float,
                cv_score_float,
                feature_names,
                Json(metadata) if metadata else None
            ))
            
            model_id = cur.fetchone()[0]
            conn.commit()
            
        logger.info(f"✅ Model uploaded successfully: {version} (ID: {model_id})")
        logger.info(f"   📊 Accuracy: {accuracy:.3f}, CV Score: {cv_score:.3f}")
        logger.info(f"   📦 Model size: {len(model_blob) / 1024:.1f} KB")
        logger.info(f"   📦 Scaler size: {len(scaler_blob) / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error uploading model: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def get_latest_model() -> Optional[Tuple[Dict, Any, Dict]]:
    """
    Download the latest active model from PostgreSQL.
    
    Returns:
        Tuple of (model_data, scaler, metadata) or None if not found
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_blob, scaler_blob, version, accuracy, cv_score, feature_names, metadata, created_at
                FROM ml_models
                WHERE is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
            """)
            
            row = cur.fetchone()
            if not row:
                logger.warning("⚠️ No active model found in database")
                return None
            
            model_blob, scaler_blob, version, accuracy, cv_score, feature_names, metadata, created_at = row
            
            # Deserialize
            model_data = joblib.load(io.BytesIO(model_blob))
            scaler = joblib.load(io.BytesIO(scaler_blob))
            
            info = {
                'version': version,
                'accuracy': accuracy,
                'cv_score': cv_score,
                'feature_names': feature_names,
                'metadata': metadata,
                'created_at': created_at.isoformat() if created_at else None
            }
            
            logger.info(f"✅ Model loaded: {version} (Accuracy: {accuracy:.3f})")
            
            return model_data, scaler, info
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None
    finally:
        conn.close()


def get_model_info() -> Optional[Dict]:
    """Get information about the latest model without downloading the blob."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT version, accuracy, cv_score, feature_names, metadata, created_at,
                       LENGTH(model_blob) as model_size,
                       LENGTH(scaler_blob) as scaler_size
                FROM ml_models
                WHERE is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
            """)
            
            row = cur.fetchone()
            if not row:
                return None
            
            return {
                'version': row[0],
                'accuracy': row[1],
                'cv_score': row[2],
                'feature_names': row[3],
                'metadata': row[4],
                'created_at': row[5].isoformat() if row[5] else None,
                'model_size_kb': row[6] / 1024 if row[6] else 0,
                'scaler_size_kb': row[7] / 1024 if row[7] else 0
            }
            
    except Exception as e:
        logger.error(f"❌ Error getting model info: {e}")
        return None
    finally:
        conn.close()


if __name__ == "__main__":
    # Test connection and table creation
    print("🔧 Initializing ML Models table...")
    init_ml_models_table()
    
    # Check for existing models
    info = get_model_info()
    if info:
        print(f"📊 Current model: {info['version']}")
        print(f"   Accuracy: {info['accuracy']:.3f}")
        print(f"   Created: {info['created_at']}")
    else:
        print("📭 No models found in database")
