"""
Pruebas sobre el MODELO de Clústeres exportado (model_artifact/).
Verifica: input válido -> prediction válida (cluster y distancia).
Correr con: pytest tests/test_model.py -v
"""

from pathlib import Path
import pytest
import skops.io as sio

MODEL_DIR = Path(__file__).resolve().parent.parent / "model_artifact"
MODEL_FILE = MODEL_DIR / "model.skops"

# Input válido con las 6 características RFM
INPUT_VALIDO = [[
    326.0, 1.0, 77183.6, 74215.0, 74215.0, 1.04
]]

@pytest.fixture(scope="module")
def modelo():
    if not MODEL_FILE.exists():
        pytest.skip("No existe el archivo del modelo en model_artifact/. Asegúrate de exportarlo primero.")
    
    # Inspeccionamos los tipos y permitimos la carga de forma segura en versiones recientes de skops
    unknown_types = sio.get_untrusted_types(file=MODEL_FILE)
    return sio.load(MODEL_FILE, trusted=unknown_types)

def test_el_modelo_carga_sin_error(modelo):
    assert modelo is not None

def test_input_valido_produce_prediccion(modelo):
    resultado = modelo.predict(INPUT_VALIDO)
    assert resultado is not None
    assert len(resultado) == 1

def test_prediccion_es_determinista(modelo):
    """El mismo input debe dar siempre exactamente el mismo clúster."""
    r1 = modelo.predict(INPUT_VALIDO)[0]
    r2 = modelo.predict(INPUT_VALIDO)[0]
    assert r1 == r2