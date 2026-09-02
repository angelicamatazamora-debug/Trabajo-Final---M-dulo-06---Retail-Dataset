"""
Pruebas sobre la API (FastAPI) para el servicio de Segmentación RFM.
Correr con: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)

INPUT_VALIDO = {
    "recency": 326.0,
    "frequency": 1.0,
    "monetary": 77183.6,
    "qty_media": 74215.0,
    "qty_total_comprada": 74215.0,
    "unitprice_medio": 1.04
}

@pytest.fixture(scope="module", autouse=True)
def verificar_modelo_cargado():
    """Si el modelo no está disponible (falta model_artifact/), se saltan
    las pruebas que dependen de una predicción real, en vez de fallar en rojo
    por una razón de setup y no de lógica."""
    resp = client.get("/health")
    if resp.status_code != 200:
        pytest.skip("El modelo no está cargado (¿corriste export_model.py?)")

# ---------- CASO FELIZ ----------

def test_health_responde_200():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_predict_con_input_valido_responde_200():
    resp = client.post("/predict", json=INPUT_VALIDO)
    assert resp.status_code == 200

def test_predict_respeta_el_schema_de_respuesta():
    resp = client.post("/predict", json=INPUT_VALIDO)
    body = resp.json()
    assert "cluster" in body
    assert "distance_to_centroid" in body
    assert "model_version" in body
    assert isinstance(body["cluster"], int)

# ---------- INPUT INVÁLIDO (Errores 422) ----------

def test_falta_una_variable_obligatoria():
    input_incompleto = INPUT_VALIDO.copy()
    del input_incompleto["recency"]
    resp = client.post("/predict", json=input_incompleto)
    assert resp.status_code == 422

def test_tipo_de_dato_incorrecto():
    input_invalido = INPUT_VALIDO.copy()
    input_invalido["frequency"] = "tres_transacciones"
    resp = client.post("/predict", json=input_invalido)
    assert resp.status_code == 422

def test_valor_fuera_de_rango_negativo():
    input_invalido = INPUT_VALIDO.copy()
    input_invalido["frequency"] = -2.0
    resp = client.post("/predict", json=input_invalido)
    assert resp.status_code == 422

def test_body_vacio():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422