import sys
import os
from pathlib import Path

# Raíz del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Librerías necesarias
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import joblib

# Importar módulos previamente definidos
from src.ingestion.Ingesta import ingest_data
from src.data_quality.quality2 import run_data_quality_pipeline
from src.engineer_features.engineer_features import construir_features_rfm

# Configuración de MLflow (una sola vez)
mlflow.set_experiment("Experimento_Clustering_RFM")


def run_training_pipeline():
    """Ejecuta el pipeline completo de entrenamiento."""
    print("--- Iniciando Pipeline de Entrenamiento ---")

    # Ingesta de datos
    print("1. Ingesta de datos...")
    df_raw = ingest_data()

    # Calidad de datos y Limpieza
    print("2. Ejecutando pipeline de calidad...")
    df_clean = run_data_quality_pipeline(df_raw)
    print(f"   Datos limpios: {df_clean.shape[0]:,} filas, {df_clean.shape[1]} columnas")

    # Crear la columna monto_total
    df_clean['monto_total'] = df_clean['quantity'] * df_clean['unitprice']

    # Feature Engineering
    print("3. Construyendo features RFM...")
    fecha_corte = df_clean['invoicedate'].max()
    df_rfm, _ = construir_features_rfm(df_clean, fecha_max_dataset=fecha_corte)
    print(f"   Matriz RFM: {df_rfm.shape[0]:,} clientes, {df_rfm.shape[1]} variables")

    # Preparación de datos para clustering
    print("4. Preparando datos para clustering...")
    vars_a_escalar = ['recency', 'frequency', 'monetary', 'qty_media', 'qty_total_comprada', 'unitprice_medio']
    X = df_rfm[vars_a_escalar].copy()

    # Transformación logarítmica
    X_log = np.log1p(X)

    # Escalado con StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    print(f"   Datos escalados: {X_scaled.shape}")

    # Entrenar K-Means
    print("5. Entrenando K-Means (k=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Calcular métricas
    sil_score = silhouette_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    print(f"   Silhouette Score: {sil_score:.4f}")
    print(f"   Calinski-Harabasz: {ch_score:.4f}")

    # MLflow tracking
    with mlflow.start_run():
        mlflow.log_param("n_clusters", 4)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("silhouette_score", sil_score)
        mlflow.log_metric("calinski_harabasz", ch_score)

        mlflow.sklearn.log_model(kmeans, "kmeans_model")

        print(" Modelo registrado en MLflow")

    # Guardar modelo y scaler localmente
    Path("models").mkdir(exist_ok=True)
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Modelo y scaler guardados en models/")


if __name__ == "__main__":
    run_training_pipeline()