#!/usr/bin/env python3
"""
Evaluación de Performance del Modelo ML
Determina si el modelo necesita retrain basado en métricas de performance
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evalúa la performance del modelo ML"""

    def __init__(self):
        self.performance_history_file = 'model_performance_history.json'
        self.min_accuracy_threshold = 0.75  # 75% accuracy mínimo
        self.max_degradation_rate = 0.05   # Máximo 5% de degradación

    def evaluate_current_model(self) -> Dict[str, Any]:
        """Evalúa el modelo actual con datos recientes"""
        try:
            # Cargar modelo actual
            import joblib
            model_data = joblib.load('nexus_system/memory_archives/ml_model.pkl')

            if isinstance(model_data, dict):
                model = model_data.get('model')
                scaler = joblib.load('nexus_system/memory_archives/scaler.pkl')
                label_encoder = model_data.get('label_encoder')
                features = model_data.get('feature_names', [])

                # Crear datos de evaluación sintéticos (en producción usar datos reales)
                eval_data = self._generate_evaluation_data()

                if eval_data is not None and len(eval_data) > 0:
                    # Preparar features
                    X_eval = eval_data[features] if all(f in eval_data.columns for f in features) else None

                    if X_eval is not None:
                        # Escalar datos
                        if scaler:
                            X_eval_scaled = scaler.transform(X_eval)
                        else:
                            X_eval_scaled = X_eval.values

                        # Hacer predicciones
                        predictions = model.predict(X_eval_scaled)

                        # Calcular métricas
                        # Para evaluación sintética, usamos distribución balanceada
                        n_samples = len(predictions)
                        y_true = np.random.choice([0, 1, 2, 3], n_samples)  # Labels sintéticas

                        accuracy = accuracy_score(y_true, predictions)

                        # Calcular confianza promedio
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(X_eval_scaled)
                            confidence_avg = np.mean(np.max(probabilities, axis=1))
                        else:
                            confidence_avg = 0.5

                        results = {
                            'timestamp': datetime.now().isoformat(),
                            'accuracy': float(accuracy),
                            'confidence_avg': float(confidence_avg),
                            'samples_evaluated': n_samples,
                            'status': 'success'
                        }

                        logger.info(".3f"                        return results
                    else:
                        logger.warning("Features no disponibles para evaluación")
                else:
                    logger.warning("No se pudieron generar datos de evaluación")

            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': 'model_not_found'
            }

        except Exception as e:
            logger.error(f"Error evaluando modelo: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }

    def _generate_evaluation_data(self) -> pd.DataFrame:
        """Genera datos sintéticos para evaluación (en producción usar datos reales)"""
        try:
            np.random.seed(42)
            n_samples = 1000

            # Crear dataframe con features similares a los de training
            data = {}
            for i in range(60):  # Aproximadamente 60 features
                if i < 20:
                    data[f'feature_{i}'] = np.random.randn(n_samples)
                elif i < 40:
                    data[f'feature_{i}'] = np.random.uniform(0, 1, n_samples)
                else:
                    data[f'feature_{i}'] = np.random.choice([-1, 0, 1], n_samples)

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error generando datos de evaluación: {e}")
            return None

    def check_performance_degradation(self) -> Tuple[bool, str]:
        """Verifica si hay degradación significativa en la performance"""
        try:
            if not os.path.exists(self.performance_history_file):
                logger.info("No hay historial de performance - primera evaluación")
                return False, "no_history"

            with open(self.performance_history_file, 'r') as f:
                history = json.load(f)

            if len(history) < 2:
                return False, "insufficient_history"

            # Obtener últimas 5 evaluaciones
            recent_evaluations = sorted(history[-5:], key=lambda x: x['timestamp'])

            accuracies = [eval['accuracy'] for eval in recent_evaluations if 'accuracy' in eval]

            if len(accuracies) < 2:
                return False, "insufficient_accuracy_data"

            # Calcular tasa de degradación
            current_accuracy = accuracies[-1]
            previous_accuracy = accuracies[-2]

            degradation = previous_accuracy - current_accuracy
            degradation_rate = degradation / previous_accuracy

            logger.info(".3f"
            # Verificar umbrales
            if current_accuracy < self.min_accuracy_threshold:
                return True, ".3f"

            if degradation_rate > self.max_degradation_rate:
                return True, ".3f"

            return False, "performance_ok"

        except Exception as e:
            logger.error(f"Error verificando degradación: {e}")
            return False, f"error: {e}"

    def save_evaluation_result(self, result: Dict[str, Any]):
        """Guarda el resultado de evaluación en el historial"""
        try:
            history = []
            if os.path.exists(self.performance_history_file):
                with open(self.performance_history_file, 'r') as f:
                    history = json.load(f)

            history.append(result)

            # Mantener solo últimas 50 evaluaciones
            if len(history) > 50:
                history = history[-50:]

            with open(self.performance_history_file, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Error guardando resultado de evaluación: {e}")

    def should_retrain_model(self) -> Tuple[bool, str]:
        """Determina si el modelo necesita retrain"""
        # Evaluar modelo actual
        current_eval = self.evaluate_current_model()
        self.save_evaluation_result(current_eval)

        if current_eval.get('status') == 'error':
            logger.warning("Evaluación falló - recomendado retrain conservativo")
            return True, "evaluation_failed"

        # Verificar degradación
        needs_retrain, reason = self.check_performance_degradation()

        return needs_retrain, reason


def main():
    """Función principal para testing"""
    evaluator = ModelEvaluator()

    print("🧪 Evaluando performance del modelo...")
    needs_retrain, reason = evaluator.should_retrain_model()

    print(f"¿Necesita retrain? {needs_retrain}")
    print(f"Razón: {reason}")

    if needs_retrain:
        print("⚠️  Se recomienda ejecutar retrain del modelo")
    else:
        print("✅ El modelo mantiene buena performance")


if __name__ == "__main__":
    main()




