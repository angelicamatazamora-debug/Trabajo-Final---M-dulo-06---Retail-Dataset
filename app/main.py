from pathlib import Path
import numpy as np
import skops.io as sio
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

# Configurar logging para el monitoreo de System (Latencia y Errores)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_monitoring")

# Ruta al directorio del modelo exportado
MODEL_DIR = Path(__file__).resolve().parent.parent / "model_artifact"
MODEL_VERSION = "1.0.0"

app = FastAPI(
    title="API de Segmentación de Clientes - Retail",
    description="API para segmentación de clientes basada en K-Means clustering (RFM).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def system_monitoring_middleware(request, call_next):
    start_time = time.time()
    
    # Procesar la petición
    response = await call_next(request)
    
    # Calcular latencia en milisegundos
    process_time = (time.time() - start_time) * 1000
    
    # Registrar métricas clave (Latencia, Status / ErrorRate implícito)
    logger.info(
        f"Method: {request.method} | Path: {request.url.path} | "
        f"Status: {response.status_code} | Latency: {process_time:.2f}ms"
    )
    
    # Inyectar la latencia en las cabeceras de respuesta para visibilidad del cliente
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    return response

def get_model():
    try:
        model_files = list(MODEL_DIR.glob("*.skops")) + list(MODEL_DIR.rglob("*.skops"))
        if model_files:
            file_path = model_files[0]
            # Obtenemos los tipos que contiene el archivo para pasarlos como confiables de forma segura
            unknown_types = sio.get_untrusted_types(file=file_path)
            model = sio.load(file_path, trusted=unknown_types)
            return model, None
        else:
            return None, "No se encontró el archivo .skops en model_artifact"
    except Exception as e:
        return None, str(e)


class CustomerFeatures(BaseModel):
    recency: float = Field(..., ge=0)
    frequency: float = Field(..., gt=0)
    monetary: float = Field(..., gt=0)
    qty_media: float = Field(..., gt=0)
    qty_total_comprada: float = Field(..., gt=0)
    unitprice_medio: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    cluster: int
    distance_to_centroid: float | None
    model_version: str

    model_config = {"protected_namespaces": ()}


@app.get("/health")
def health_check():
    model, load_error = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {load_error}")
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    model, load_error = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail=f"El modelo no se cargó correctamente: {load_error}")
    
    try:
        raw_vector = [[
            features.recency,
            features.frequency,
            features.monetary,
            features.qty_media,
            features.qty_total_comprada,
            features.unitprice_medio
        ]]
        
        X_log = np.log1p(raw_vector)
        
        # Predicción directa con el pipeline cargado mediante skops
        pred = model.predict(X_log)
        cluster = int(np.ravel(pred)[0])

        # Extracción de pasos del pipeline de sklearn
        X_scaled = model.named_steps['scaler'].transform(X_log)
        centroids = model.named_steps['kmeans'].cluster_centers_
        assigned_centroid = centroids[cluster]
        
        distance = float(np.linalg.norm(X_scaled[0] - assigned_centroid))
        distance_rounded = round(distance, 4)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la inferencia: {str(e)}")

    return {
        "cluster": cluster,
        "distance_to_centroid": distance_rounded,
        "model_version": MODEL_VERSION,
    }

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API de Segmentación de Clientes - Retail online"}