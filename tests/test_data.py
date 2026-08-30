"""
Pruebas sobre los DATOS de entrada del modelo de Segmentación RFM.
Cubre: esquema, tipos, rangos, missing values, variables obligatorias.
Correr con: pytest tests/test_data.py -v
"""

import numpy as np
import pandas as pd
import pytest

# Columnas exactas que tu modelo RFM espera recibir
COLUMNAS_ESPERADAS = [
    "recency", "frequency", "monetary", 
    "qty_media", "qty_total_comprada", "unitprice_medio"
]

# Rangos basados en la lógica de negocio y validación del modelo
RANGOS_ESPERADOS = {
    "recency": (0, 2000),         # Días desde la última compra
    "frequency": (1, 10000),      # Cantidad de transacciones
    "monetary": (0, 1000000),     # Monto total gastado ($)
    "qty_media": (0, 1000000),    # Cantidad media por transacción
    "qty_total_comprada": (0, 5000000), # Cantidad total acumulada
    "unitprice_medio": (0, 10000) # Precio unitario promedio
}

@pytest.fixture(scope="module")
def df_muestra():
    """Crea un DataFrame de prueba sintético o basado en los inputs habituales."""
    data = {
        "recency": [326.0, 15.0, 50.0],
        "frequency": [1.0, 12.0, 4.0],
        "monetary": [77183.6, 1500.0, 4500.0],
        "qty_media": [74215.0, 5.0, 20.0],
        "qty_total_comprada": [74215.0, 60.0, 80.0],
        "unitprice_medio": [1.04, 300.0, 56.2]
    }
    return pd.DataFrame(data)

# ---------- ESQUEMA ----------

def test_columnas_presentes(df_muestra):
    assert list(df_muestra.columns) == COLUMNAS_ESPERADAS

def test_no_hay_columnas_extra(df_muestra):
    assert set(df_muestra.columns) == set(COLUMNAS_ESPERADAS)

# ---------- TIPOS ----------

def test_todas_las_columnas_son_numericas(df_muestra):
    for col in df_muestra.columns:
        assert pd.api.types.is_numeric_dtype(df_muestra[col]), f"{col} no es numérica"

# ---------- RANGOS ----------

@pytest.mark.parametrize("col", COLUMNAS_ESPERADAS)
def test_rangos_por_columna(df_muestra, col):
    minimo, maximo = RANGOS_ESPERADOS[col]
    assert df_muestra[col].min() >= minimo, f"{col} tiene valores por debajo de {minimo}"
    assert df_muestra[col].max() <= maximo, f"{col} tiene valores por encima de {maximo}"

def test_no_hay_valores_negativos(df_muestra):
    for col in COLUMNAS_ESPERADAS:
        assert (df_muestra[col] >= 0).all(), f"{col} tiene valores negativos"

# ---------- MISSING ----------

def test_no_hay_valores_faltantes(df_muestra):
    assert df_muestra.isnull().sum().sum() == 0, "El dataset no debería tener valores nulos"

def test_no_hay_infinitos(df_muestra):
    assert np.isfinite(df_muestra.values).all(), "El dataset no debería tener valores infinitos"

# ---------- VARIABLES OBLIGATORIAS ----------

@pytest.mark.parametrize("col", COLUMNAS_ESPERADAS)
def test_variable_obligatoria_presente(df_muestra, col):
    assert col in df_muestra.columns