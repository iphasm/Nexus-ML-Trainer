# Configuración Óptima de Datos y Retrain para ML Trading

## 📊 Configuración de Datos Actual

### Parámetros Optimizados
- **Velas por símbolo**: 12,000 (antes: 15,000)
- **Intervalo**: 15 minutos
- **Ventana temporal**: ~4.5 meses de datos históricos
- **Total velas**: ~4,000-5,000 velas procesables (después de limpieza)

### ¿Por qué 12,000 velas?
- ✅ **Suficiente histórico**: Cubre ciclos de mercado completos
- ✅ **Datos relevantes**: Los más recientes son más predictivos
- ✅ **Tiempo de entrenamiento**: Optimizado (no excesivo)
- ✅ **Memoria eficiente**: Manejable en entornos cloud

## 🔄 Sistema de Retrain Automático

### Triggers de Retrain
1. **Tiempo máximo**: Cada 7 días (fuerza retrain)
2. **Intervalo regular**: Cada 24 horas (evaluación programada)
3. **Performance**: Si accuracy < 75% o degradación > 5%
4. **Manual**: Después de eventos importantes de mercado

### Archivos del Sistema
- `auto_retrain.py` - Gestor principal de retrain automático
- `model_evaluation.py` - Evaluador de performance del modelo
- `schedule_retrain.sh` - Script para cron/scheduling

### Uso del Sistema

#### Configuración de Variables de Entorno
```bash
# Frecuencia de verificación (horas)
export RETRAIN_CHECK_HOURS=24

# Umbral mínimo de accuracy
export PERFORMANCE_THRESHOLD=0.75

# Máximo días sin retrain forzado
export FORCE_RETRAIN_DAYS=7
```

#### Scheduling con Cron
```bash
# Verificar cada 6 horas
0 */6 * * * /path/to/schedule_retrain.sh

# Verificar diariamente a las 2 AM
0 2 * * * /path/to/schedule_retrain.sh
```

#### Ejecución Manual
```bash
# Verificar si necesita retrain
python auto_retrain.py --check-only

# Forzar retrain inmediato
python auto_retrain.py --force --verbose

# Ejecutar evaluación normal
python auto_retrain.py
```

## 📈 Estrategias de Datos por Mercado

### Crypto (Recomendado Actual)
- **Velas**: 12,000 (4.5 meses)
- **Retrain**: Cada 2-7 días
- **Razón**: Mercado volátil, cambia rápidamente

### Forex/Stocks (Si se expande)
- **Velas**: 16,000-20,000 (6-7.5 meses)
- **Retrain**: Semanal/mensual
- **Razón**: Mercados más estables

## 🎯 Recomendaciones de Implementación

### Fase 1: Configuración Básica
1. ✅ **Datos optimizados**: 12,000 velas implementado
2. ✅ **Retrain automático**: Sistema completo implementado
3. ✅ **Evaluación de performance**: Sistema de monitoreo implementado

### Fase 2: Producción
1. **Configurar cron job** en el servidor cloud
2. **Monitorear logs** de retrain automático
3. **Ajustar umbrales** basado en performance real
4. **Alertas**: Configurar notificaciones de errores

### Fase 3: Optimización Continua
1. **A/B Testing**: Probar diferentes ventanas de datos
2. **Feature Importance**: Remover features poco útiles
3. **Model Selection**: Probar otros algoritmos (LightGBM, CatBoost)
4. **Ensemble**: Combinar múltiples modelos

## 📊 Métricas de Monitoreo

### Performance del Modelo
- **Accuracy**: > 75% objetivo mínimo
- **Degradación**: < 5% máximo por período
- **Confianza**: > 0.6 promedio en predicciones

### Sistema de Retrain
- **Frecuencia**: Logs de cuándo se ejecuta
- **Éxito**: Tasa de retrain exitosos
- **Tiempo**: Duración promedio de retrain

## 🚨 Alertas y Monitoreo

### Condiciones de Alerta
- Retrain falla por 3+ veces consecutivas
- Accuracy cae por debajo del 70%
- Tiempo de retrain > 2 horas
- Error en evaluación de modelo

### Logs Importantes
```
auto_retrain.log          # Logs del sistema de retrain
model_performance_history.json  # Historial de performance
retrain_status.json       # Estado del último retrain
```

## 🎛️ Configuración Avanzada

### Variables de Entorno Detalladas
```bash
# Sistema de retrain
RETRAIN_CHECK_HOURS=24          # Verificar cada 24 horas
PERFORMANCE_THRESHOLD=0.75      # Accuracy mínimo
FORCE_RETRAIN_DAYS=7           # Máximo sin retrain
MAX_RETRAIN_DURATION=7200      # Timeout 2 horas

# Evaluación de modelo
EVALUATION_SAMPLES=1000        # Muestras para evaluación
PERFORMANCE_HISTORY_SIZE=50    # Mantener 50 evaluaciones
DEGRADATION_WINDOW=5           # Últimas 5 evaluaciones para análisis
```

### Optimizaciones de Performance
- **Paralelización**: Usar múltiples cores para training
- **Cache**: Almacenar features preprocesados
- **Incremental**: Actualizar modelo en lugar de retrain completo
- **Early Stopping**: Detener training cuando no mejora

## 📋 Checklist de Implementación

- [x] Configurar ventana de datos óptima (12k velas)
- [x] Implementar sistema de retrain automático
- [x] Crear evaluador de performance
- [x] Configurar scheduling script
- [ ] Configurar cron job en producción
- [ ] Probar sistema en staging
- [ ] Monitorear primera semana de operación
- [ ] Ajustar umbrales basado en datos reales

---

**Resultado**: Sistema de ML con mantenimiento automático, optimizado para crypto markets con retrain inteligente basado en performance real.




