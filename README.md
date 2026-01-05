# Nexus ML Trainer

Worker de entrenamiento de Machine Learning para el [Nexus Trading Bot](https://github.com/iphasm/Nexus-TB).

## Arquitectura

Este servicio se ejecuta de forma independiente al bot principal y se encarga de:

1. **Descargar datos históricos** de Binance/Bybit.
2. **Calcular indicadores técnicos** y generar etiquetas para entrenamiento.
3. **Entrenar el modelo XGBoost** con los datos procesados.
4. **Guardar el modelo en PostgreSQL** para que el bot principal lo descargue.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAILWAY PROJECT                                 │
│                                                                         │
│  ┌───────────────────────────┐     ┌───────────────────────────┐       │
│  │   nexus-bot (Servicio 1)  │     │  ml-trainer (Servicio 2)  │       │
│  │  - Deploy: Nexus-TB       │     │  - Deploy: Nexus-ML-Trainer│      │
│  │  - Runtime: 24/7          │     │  - Runtime: Cron cada 24h │       │
│  └─────────────┬─────────────┘     └─────────────┬─────────────┘       │
│                │                                 │                      │
│                ▼                                 ▼                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    PostgreSQL (Compartida)                      │   │
│  │  - ml_models (tabla con el .pkl serializado)                    │   │
│  │  - training_logs (historial de entrenamientos)                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Variables de Entorno Requeridas

```env
DATABASE_URL=postgresql://user:password@host:port/dbname
BINANCE_API_KEY=your_binance_api_key (opcional, para datos públicos no es necesario)
BINANCE_API_SECRET=your_binance_api_secret (opcional)
```

## Ejecución Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar entrenamiento
python train_cortex.py --candles 15000 --symbols 10
```

## Despliegue en Railway

1. Conectar este repositorio a Railway.
2. Configurar las variables de entorno (`DATABASE_URL`).
3. Opcional: Configurar un Cron Job para ejecutar cada 24 horas.

## Archivos Principales

| Archivo | Descripción |
|---------|-------------|
| `train_cortex.py` | Script principal de entrenamiento |
| `model_uploader.py` | Guarda el modelo en PostgreSQL |
| `indicators.py` | Cálculo de indicadores técnicos |
| `Dockerfile` | Imagen para Railway |
| `railway.json` | Configuración de despliegue |

## Comunicación con Nexus-TB

El modelo entrenado se guarda en la tabla `ml_models` de PostgreSQL como un blob (`bytea`). El bot principal (`Nexus-TB`) descarga el modelo al iniciar o cuando se ejecuta el comando `/reload_model`.

### Esquema de la tabla `ml_models`:

```sql
CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    model_blob BYTEA NOT NULL,
    scaler_blob BYTEA NOT NULL,
    accuracy FLOAT,
    cv_score FLOAT,
    feature_names TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);
```

## Licencia

MIT
