# Importar librerías
import sys
import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Importar módulos previamente definidos
from src.ingestion.ingest import ingest_data
from src.quality.quality import run_data_quality_pipeline
from src.engineer_features.engineer_features import construir_features_rfm
from sklearn.pipeline import Pipeline

# Conexión al servidor de MLFlow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("online-retail-mlflow")

def registrar_corrida_modelo(nombre_modelo, pipeline, X_log, df_rfm, fecha_corte, vars_a_escalar):
    temp_dir = Path("temp_artifacts")
    temp_dir.mkdir(exist_ok=True)

    with mlflow.start_run(run_name=nombre_modelo):
        # Ajustar el pipeline completo (escala los datos y ejecuta el modelo)
        labels = pipeline.fit_predict(X_log)
        
        # Extraer la matriz ya escalada internamente para métricas y PCA
        X_scaled = pipeline.named_steps['scaler'].transform(X_log)
        
        n_clusters_efectivos = len(set(labels)) - (1 if -1 in labels else 0)
        sil_score = silhouette_score(X_scaled, labels) if n_clusters_efectivos > 1 else -1
        ch_score = calinski_harabasz_score(X_scaled, labels) if n_clusters_efectivos > 1 else 0

        # Parámetros y Métricas
        mlflow.log_param("algorithm", nombre_modelo)
        mlflow.log_param("feature_set", str(vars_a_escalar))
        mlflow.log_param("random_seed", 42)
        mlflow.log_param("data_version", str(fecha_corte))
        mlflow.log_param("n_customers", df_rfm.shape[0])

        mlflow.log_metric("silhouette_score", sil_score)
        mlflow.log_metric("calinski_harabasz_score", ch_score)

        # Registrar el Pipeline completo en MLflow (guarda scaler + modelo juntos)
        mlflow.sklearn.log_model(pipeline, name="model")

        # Generar imagen con PCA sobre los datos escalados
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.6)
        ax.set_title(f"Cluster Analysis - {nombre_modelo}")
        plt.colorbar(scatter, label="Clúster")
        
        safe_name = nombre_modelo.lower().replace(' ', '_').replace('=', '').replace('(', '').replace(')', '').replace(',', '').replace('.', '_')
        plot_path = temp_dir / f"cluster_analysis_{safe_name}.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(plot_path))

        # Generar JSON de configuración
        config = {
            "algorithm": nombre_modelo,
            "feature_set": vars_a_escalar,
            "data_version": str(fecha_corte),
            "metrics": {"silhouette_score": sil_score, "calinski_harabasz_score": ch_score}
        }
        config_path = temp_dir / "run_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(str(config_path))

        print(f"-> Run '{nombre_modelo}' registrado con éxito en MLflow.")

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

def ejecutar_experimentos(X_log, df_rfm, fecha_corte, vars_a_escalar):
    modelos_clustering = {
        "K-Means (k=4)": Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=4, random_state=42, n_init=10))
        ]),
        "Clustering Jerárquico (Ward, k=4)": Pipeline([
            ('scaler', StandardScaler()),
            ('ward', AgglomerativeClustering(n_clusters=4, linkage='ward'))
        ]),
        "DBSCAN (eps=1.5, min_samples=10)": Pipeline([
            ('scaler', StandardScaler()),
            ('dbscan', DBSCAN(eps=1.5, min_samples=10))
        ])
    }

    for nombre, pipeline in modelos_clustering.items():
        registrar_corrida_modelo(nombre, pipeline, X_log, df_rfm, fecha_corte, vars_a_escalar)

if __name__ == "__main__":
    print("--- Iniciando Experimentos en MLflow ---")
    
    # 1. Cargar y preparar datos
    df_raw = ingest_data()
    df_clean = run_data_quality_pipeline(df_raw)
    
    df_clean['monto_total'] = df_clean['quantity'] * df_clean['unitprice']
    fecha_corte = df_clean['invoicedate'].max()
    df_rfm, _ = construir_features_rfm(df_clean, fecha_max_dataset=fecha_corte)

    vars_a_escalar = ['recency', 'frequency', 'monetary', 'qty_media', 'qty_total_comprada', 'unitprice_medio']
    X = df_rfm[vars_a_escalar].copy()
    
    X_log = np.log1p(X)

    # 2. Ejecutar los experimentos separados para los 3 modelos
    ejecutar_experimentos(X_log, df_rfm, fecha_corte, vars_a_escalar)