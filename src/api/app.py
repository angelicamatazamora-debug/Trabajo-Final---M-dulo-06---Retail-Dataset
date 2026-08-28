from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
import mlflow.sklearn
import joblib
from pathlib import Path
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.api.schemas import CustomerFeatures, PredictionResponse, HealthResponse

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Segmentación de Clientes - Retail",
    description="""
    API para segmentación de clientes basada en comportamiento de compra.

    ## Modelo utilizado:
    - Algoritmo: K-Means clustering
    - Número de clusters: 4
    - Features: RFM enriquecido (6 variables)
    """,
    version="1.0.0"
)

# Cargar el modelo y el scaler
MODEL_NAME = "kmeans_rfm_clustering"

# Intentar cargar desde MLflow
try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    logger.info(f" Modelo {MODEL_NAME} cargado desde MLflow")
except Exception as e:
    logger.warning(f"No se pudo cargar desde MLflow: {e}")
    # Fallback: cargar desde archivo local
    model_path = Path("models") / "kmeans_model.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
        logger.info(f" Modelo cargado desde {model_path}")
    else:
        model = None
        logger.error(" No se encontró el modelo")

# Cargar scaler
scaler_path = Path("models") / "scaler.pkl"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    logger.info("Scaler cargado")
else:
    scaler = None
    logger.warning("Scaler no encontrado")

# Endpoints

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Verifica el estado de la API y si el modelo está cargado."""
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        message="Servicio operativo" if model is not None else "Modelo no disponible",
        model_loaded=model is not None,
        model_version=MODEL_NAME if model is not None else None
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    """
    Predice el cluster para un solo cliente.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        # Convertir a DataFrame
        input_df = pd.DataFrame([features.dict()])

        # Aplicar log transform (igual que en entrenamiento)
        input_log = np.log1p(input_df)

        # Escalar
        if scaler is None:
            raise HTTPException(status_code=500, detail="Scaler no disponible")
        input_scaled = scaler.transform(input_log)

        # Predecir
        cluster = model.predict(input_scaled)[0]
        distances = model.transform(input_scaled)
        distance_to_centroid = float(distances[0][cluster])

        return PredictionResponse(
            cluster=int(cluster),
            distance_to_centroid=distance_to_centroid,
            model_version=MODEL_NAME,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# --- Punto de entrada para ejecución directa ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)