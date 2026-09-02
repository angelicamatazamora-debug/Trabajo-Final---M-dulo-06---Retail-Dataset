import shutil
from pathlib import Path
import mlflow

# ---------------------------------------------
TRACKING_URI = "http://127.0.0.1:5000"
MODEL_URI = "models:/kmeans_retail_model@production" 
OUTPUT_DIR = "model_artifact"
# ---------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)

output_path = Path(OUTPUT_DIR)
if output_path.exists():
    shutil.rmtree(output_path) 

print(f"Descargando artefacto del modelo desde: {MODEL_URI}")
# Descarga el directorio completo registrado en el servidor de MLflow sin asumir formato sklearn
artifact_path = mlflow.artifacts.download_artifacts(artifact_uri=MODEL_URI)

# Copia el contenido descargado a tu carpeta local model_artifact/
shutil.copytree(artifact_path, output_path)

print(f"Guardando modelo autocontenido en: {OUTPUT_DIR}/")
print("El contenedor ya NO necesitará conectarse a tu mlflow server para predecir.")