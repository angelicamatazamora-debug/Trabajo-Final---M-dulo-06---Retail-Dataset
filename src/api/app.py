#Importar librerías

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import uvicorn
import os
from datetime import datetime
import logging

# Importar los esquemas de datos
from src.api.schemas import (
    CustomerFeatures,
    PredictionResponse,
    HealthResponse,
    BatchPredictionResponse
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#FastAPI app
app = FastAPI(
    title="API de Segmentación de Clientes - Retail",
    description="""
    API para segmentación de clientes basada en comportamiento de compra.
    
    ## Modelo utilizado:
    - Algoritmo: K-Means clustering
    - Número de clusters: 4
    - Features: RFM enriquecido (6 variables)
    """,
    version="1.0.0",
    contact={
        "name": "Equipo 5 - Retail Dataset",
        "email": "equipo5@retail.com"
    }
)


