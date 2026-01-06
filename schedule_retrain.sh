#!/bin/bash
# Script para programar retrain automático del modelo ML
# Uso: ./schedule_retrain.sh
# O agregar a crontab: 0 */6 * * * /path/to/schedule_retrain.sh

echo "🤖 Auto Retrain Scheduler - $(date)"
echo "==================================="

# Directorio del proyecto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Activar entorno virtual si existe
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Entorno virtual activado"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Entorno virtual activado (.venv)"
else
    echo "⚠️  No se encontró entorno virtual"
fi

# Ejecutar evaluación y retrain automático
echo "🔍 Ejecutando evaluación del modelo..."
python auto_retrain.py

# Verificar resultado
if [ $? -eq 0 ]; then
    echo "✅ Proceso de auto retrain completado exitosamente"
else
    echo "❌ Error en el proceso de auto retrain"
    # Enviar notificación de error (opcional)
    # curl -X POST -H 'Content-type: application/json' \
    #      --data '{"text":"Error en auto retrain del modelo ML"}' \
    #      YOUR_WEBHOOK_URL
fi

echo "🏁 Script completado - $(date)"
echo ""
