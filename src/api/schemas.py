# Importar Librerías iniciales
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
#Esquema de entrada para la predicción de segmentación de clientes
class CustomerFeatures(BaseModel):
    """
    Esquema de entrada para la predicción de segmentación de clientes
    """
    recency: int = Field(..., description="Días desde la última compra")
    frequency: int = Field(..., description="Número de compras realizadas")
    monetary: float = Field(..., description="Gasto total acumulado")
    qty_media: float = Field(..., description="Cantidad promedio por compra")
    qty_total_comprada: int = Field(..., description="Cantidad total de productos comprados")
    unitprice_medio: float = Field(..., description="Precio promedio por unidad")
    
    class Config:
        json_schema_extra = {
            "example": {
                "recency": 30,
                "frequency": 5,
                "monetary": 1500.50,
                "qty_media": 12.5,
                "qty_total_comprada": 2458,
                "unitprice_medio": 2.64
            }
        }

#Esquema de salida para la predicción de segmentación de clientes
class PredictionResponse(BaseModel):
    """
    Respuesta de la predicción.
    """
    cluster: int = Field(..., description="Número del cluster asignado")
    distance_to_centroid: float = Field(..., description="Distancia al centroide del cluster")
    model_version: str = Field(..., description="Versión del modelo utilizado")
    timestamp: str = Field(..., description="Timestamp de la predicción")
    cluster_profile: Optional[dict] = Field(None, description="Perfil del cluster")

class HealthResponse(BaseModel):
    """Verifica el estado de salud del servicio."""
    status: str
    message: str
    model_loaded: bool
    model_version: Optional[str] = None 