import pandas as pd
from pathlib import Path
from src.monitoring.monitor import evaluate_data_drift, evaluate_model_monitoring
from app.main import get_model

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Cargar el modelo entrenado
model, error = get_model()
if model is None:
    raise RuntimeError(f"No se pudo cargar el modelo: {error}")

# 2. Cargar los datos limpios transaccionales
DATA_DIR = Path("data")
df_transactions = pd.read_csv(DATA_DIR / "processed" / "online_retail_clean.csv")

# 3. Función para transformar transacciones en el formato RFM que espera el modelo
def create_rfm_features(df):
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    snapshot_date = df['invoicedate'].max() + pd.Timedelta(days=1)
    df['total_price'] = df['quantity'] * df['unitprice']
    
    rfm = df.groupby('customerid').agg({
        'invoicedate': lambda x: (snapshot_date - x.max()).days, # recency
        'invoiceno': 'nunique',                                    # frequency
        'total_price': 'sum',                                      # monetary
        'quantity': ['mean', 'sum'],                               # qty_media, qty_total_comprada
        'unitprice': 'mean'                                        # unitprice_medio
    })
    
    rfm.columns = ['recency', 'frequency', 'monetary', 'qty_media', 'qty_total_comprada', 'unitprice_medio']
    return rfm.reset_index().dropna()

# Generar el dataset agregado de clientes
print("Transformando transacciones a formato RFM por cliente...")
df_full_rfm = create_rfm_features(df_transactions)

# Simular división entre referencia (entrenamiento) y lote de producción
df_ref = df_full_rfm.sample(frac=0.7, random_state=42)
df_prod = df_full_rfm.drop(df_ref.index).sample(n=min(1000, len(df_full_rfm) - len(df_ref)), random_state=42)

features = ["recency", "frequency", "monetary", "qty_media", "qty_total_comprada", "unitprice_medio"]

# 4. Ejecutar Monitoreo de Datos (O2 - PSI)
print("\n--- RESULTADOS DATA MONITORING (PSI) ---")
drift_report = evaluate_data_drift(df_ref, df_prod, features)
for var, res in drift_report.items():
    print(f"Variable: {var} | PSI: {res['psi']} | Estado: {res['status']}")

# 5. Ejecutar Monitoreo de Modelo (O3 - Clustering)
print("\n--- RESULTADOS MODEL MONITORING (Clustering) ---")
model_report = evaluate_model_monitoring(model, df_prod[features])
print("Distribución de Clústeres (Conteo):", model_report["cluster_distribution_counts"])
print("Distribución de Clústeres (%):", model_report["cluster_distribution_proportions"])
print("Distancia promedio a centroides:", model_report["avg_distance_to_centroid"])