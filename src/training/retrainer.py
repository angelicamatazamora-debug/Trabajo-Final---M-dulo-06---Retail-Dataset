import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RetrainerLogger")

def retrain_kmeans_model(new_data_path: str, model_output_path: str = "model_artifact/kmeans_model.pkl", n_clusters: int = 4):
    """
    Ejecuta el reentrenamiento del modelo K-Means utilizando datos nuevos validados,
    actualizando el artefacto en producción.
    """
    logger.info("Iniciando proceso de reentrenamiento del modelo...")
    
    # 1. Cargar datos nuevos
    df = pd.read_csv(new_data_path)
    
    # 2. Transformar a formato RFM (reutilizando la lógica base del proyecto)
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    snapshot_date = df['invoicedate'].max() + pd.Timedelta(days=1)
    df['total_price'] = df['quantity'] * df['unitprice']
    
    rfm = df.groupby('customerid').agg({
        'invoicedate': lambda x: (snapshot_date - x.max()).days,
        'invoiceno': 'nunique',
        'total_price': 'sum',
        'quantity': ['mean', 'sum'],
        'unitprice': 'mean'
    })
    rfm.columns = ['recency', 'frequency', 'monetary', 'qty_media', 'qty_total_comprada', 'unitprice_medio']
    rfm = rfm.reset_index().dropna()
    
    # Seleccionar features de entrenamiento
    features = ["recency", "frequency", "monetary", "qty_media", "qty_total_comprada", "unitprice_medio"]
    X = rfm[features]
    
    # 3. Entrenar nueva versión del K-Means
    logger.info(f"Entrenando K-Means con {len(X)} registros de clientes y {n_clusters} clústeres...")
    new_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    new_model.fit(X)
    
    # 4. Guardar artefacto actualizado
    output_path = Path(model_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(new_model, output_path)
    
    logger.info(f"Modelo reentrenado y guardado exitosamente en: {output_path}")
    return {"status": "SUCCESS", "model_path": str(output_path), "samples_used": len(X)}