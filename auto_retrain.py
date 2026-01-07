#!/usr/bin/env python3
"""
Sistema de Retrain Automático para el Modelo ML
Ejecuta retrain periódico basado en evaluación de performance
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(level)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_retrain.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutoRetrainManager:
    """Gestiona el retrain automático del modelo ML"""

    def __init__(self):
        self.check_interval_hours = int(os.getenv('RETRAIN_CHECK_HOURS', '24'))  # 24 horas por defecto
        self.performance_threshold = float(os.getenv('PERFORMANCE_THRESHOLD', '0.75'))  # 75% accuracy mínimo
        self.force_retrain_days = int(os.getenv('FORCE_RETRAIN_DAYS', '7'))  # Máximo 7 días sin retrain

        # Archivo para trackear último retrain
        self.status_file = 'retrain_status.json'

    def should_retrain(self) -> tuple[bool, str]:
        """
        Determina si se debe hacer retrain basado en:
        1. Tiempo desde último retrain
        2. Performance del modelo actual
        3. Condiciones de mercado
        """
        try:
            import json
            from pathlib import Path

            # Verificar archivo de status
            status_path = Path(self.status_file)
            if not status_path.exists():
                logger.info("No hay registro de retrain anterior - ejecutando retrain inicial")
                return True, "retrain_initial"

            # Leer status anterior
            with open(status_path, 'r') as f:
                status = json.load(f)

            last_retrain = datetime.fromisoformat(status.get('last_retrain', '2020-01-01T00:00:00'))
            hours_since_retrain = (datetime.now() - last_retrain).total_seconds() / 3600

            # Verificar tiempo máximo sin retrain
            max_hours = self.force_retrain_days * 24
            if hours_since_retrain > max_hours:
                logger.info(".1f"                return True, f"max_time_exceeded_{self.force_retrain_days}d"

            # Verificar intervalo regular
            if hours_since_retrain >= self.check_interval_hours:
                # Evaluar performance del modelo actual
                performance_ok = self._evaluate_model_performance()

                if not performance_ok:
                    logger.warning(".2f"                    return True, "performance_degraded"

                logger.info("Retrain periódico - modelo funcionando bien")
                return True, "scheduled_retrain"

            logger.info(".1f"            return False, "too_soon"

        except Exception as e:
            logger.error(f"Error evaluando necesidad de retrain: {e}")
            # En caso de error, hacer retrain conservativo
            return True, "error_fallback"

    def _evaluate_model_performance(self) -> bool:
        """Evalúa si el modelo actual tiene performance aceptable"""
        try:
            # Intentar importar métricas de performance
            # Esto podría incluir: accuracy reciente, win rate, etc.
            # Por ahora, retornar True (asumir performance buena)
            return True
        except Exception:
            return False

    def execute_retrain(self, verbose: bool = False) -> bool:
        """Ejecuta el proceso completo de retrain"""
        try:
            import json
            from pathlib import Path

            logger.info("🚀 Iniciando proceso de retrain automático")

            # Ejecutar el training
            from train_cortex import train

            # Usar configuración optimizada para crypto
            symbols = None  # Usar todos los símbolos por defecto
            max_candles = 12000  # 4.5 meses optimizados

            logger.info(f"Entrenando con {max_candles} velas por símbolo")

            # Ejecutar training
            train(symbols=symbols, max_candles=max_candles, verbose=verbose)

            # Actualizar status
            status = {
                'last_retrain': datetime.now().isoformat(),
                'candles_used': max_candles,
                'status': 'success',
                'next_check': (datetime.now() + timedelta(hours=self.check_interval_hours)).isoformat()
            }

            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)

            logger.info("✅ Retrain completado exitosamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error durante retrain: {e}")

            # Actualizar status con error
            try:
                import json
                status = {
                    'last_retrain': datetime.now().isoformat(),
                    'status': 'error',
                    'error': str(e),
                    'next_check': (datetime.now() + timedelta(hours=1)).isoformat()  # Reintentar en 1 hora
                }
                with open(self.status_file, 'w') as f:
                    json.dump(status, f, indent=2)
            except:
                pass

            return False

    def run_auto_retrain(self, verbose: bool = False) -> bool:
        """Método principal para ejecutar retrain automático"""
        logger.info("🤖 Verificando necesidad de retrain automático")

        should_run, reason = self.should_retrain()

        if should_run:
            logger.info(f"📊 Ejecutando retrain por: {reason}")
            success = self.execute_retrain(verbose=verbose)

            if success:
                logger.info("🎉 Retrain automático completado exitosamente")
                return True
            else:
                logger.error("💥 Retrain automático falló")
                return False
        else:
            logger.info(f"⏳ No se necesita retrain: {reason}")
            return True


def main():
    """Función principal para ejecutar desde línea de comandos"""
    import argparse

    parser = argparse.ArgumentParser(description='Auto Retrain Manager para Modelo ML')
    parser.add_argument('--verbose', '-v', action='store_true', help='Modo verbose')
    parser.add_argument('--force', '-f', action='store_true', help='Forzar retrain inmediato')
    parser.add_argument('--check-only', action='store_true', help='Solo verificar si se necesita retrain')

    args = parser.parse_args()

    manager = AutoRetrainManager()

    if args.force:
        logger.info("🔧 Forzando retrain inmediato")
        success = manager.execute_retrain(verbose=args.verbose)
    elif args.check_only:
        should_run, reason = manager.should_retrain()
        print(f"Should retrain: {should_run} (reason: {reason})")
        return
    else:
        success = manager.run_auto_retrain(verbose=args.verbose)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()




