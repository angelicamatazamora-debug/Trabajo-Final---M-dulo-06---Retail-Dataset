import numpy as np
import pandas as pd

def calculate_psi(reference: pd.Series, production: pd.Series, bins: int = 10) -> float:
    """Calcula el Population Stability Index (PSI) para datos numéricos."""
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(reference.dropna(), quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference.dropna(), bins=bin_edges)
    prod_counts, _ = np.histogram(production.dropna(), bins=bin_edges)

    ref_pct = ref_counts / len(reference.dropna())
    prod_pct = prod_counts / len(production.dropna())

    eps = 1e-4
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    prod_pct = np.where(prod_pct == 0, eps, prod_pct)

    psi_value = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi_value)

def evaluate_data_drift(df_reference: pd.DataFrame, df_production: pd.DataFrame, features: list) -> dict:
    """O2: Data Monitoring evaluando PSI por variable."""
    drift_results = {}
    for col in features:
        if col in df_reference.columns and col in df_production.columns:
            psi = calculate_psi(df_reference[col], df_production[col])
            
            # Definición de umbrales justificados
            if psi < 0.1:
                status = "OK"
            elif 0.1 <= psi <= 0.25:
                status = "WARNING"
            else:
                status = "ALERT"
                
            drift_results[col] = {"psi": round(psi, 4), "status": status}
    return drift_results

def evaluate_model_monitoring(model, df_production_features: pd.DataFrame) -> dict:
    """O3: Model Monitoring para Clustering (Distribución y Centroides)."""
    # Transformación y predicción con el pipeline
    X_log = np.log1p(df_production_features)
    preds = model.predict(X_log)
    
    # 1. Distribución de clústeres en producción (%)
    unique_clusters, counts = np.unique(preds, return_counts=True)
    cluster_distribution = {int(c): int(cnt) for c, cnt in zip(unique_clusters, counts)}
    cluster_proportions = {int(c): round(cnt / len(preds), 4) for c, cnt in zip(unique_clusters, counts)}
    
    # 2. Desplazamiento de centroides (Distancia promedio al centroide asignado)
    X_scaled = model.named_steps['scaler'].transform(X_log)
    centroids = model.named_steps['kmeans'].cluster_centers_
    
    distances = []
    for i, cluster_id in enumerate(preds):
        assigned_centroid = centroids[cluster_id]
        dist = np.linalg.norm(X_scaled[i] - assigned_centroid)
        distances.append(dist)
        
    avg_centroid_distance = float(np.mean(distances))
    
    return {
        "cluster_distribution_counts": cluster_distribution,
        "cluster_distribution_proportions": cluster_proportions,
        "avg_distance_to_centroid": round(avg_centroid_distance, 4)
    }