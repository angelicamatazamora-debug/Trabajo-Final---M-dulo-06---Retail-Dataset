import pandas as pd
import numpy as np
from datetime import timedelta

def construir_features_rfm(df_transaccional, fecha_max_dataset=None):
    """
    Función unificada y reutilizable de ingeniería de características RFM.
    Garantiza que la misma lógica se aplique tanto en experimentación (Notebook) 
    como en despliegue (Producción/API), evitando Data Leakage.
    """
    df = df_transaccional.copy()
    
    # Si no se provee una fecha máxima (en producción), se toma la máxima del lote actual
    if fecha_max_dataset is None:
        fecha_referencia = df['invoicedate'].max() + timedelta(days=1)
    else:
        fecha_referencia = fecha_max_dataset + timedelta(days=1)
        
    # Agregación a nivel de cliente
    df_rfm = df.groupby('customerid').agg({
        'invoicedate': lambda x: (fecha_referencia - x.max()).days,
        'invoiceno': 'nunique',
        'monto_total': 'sum'
    }).rename(columns={
        'invoicedate': 'recency',
        'invoiceno': 'frequency',
        'monto_total': 'monetary'
    }).reset_index()
    
    # Variables adicionales de comportamiento para enriquecer el modelo (más allá del RFM básico)
    df_aux = df.groupby('customerid').agg({
        'quantity': ['mean', 'sum'],
        'unitprice': 'mean'
    }).reset_index()
    df_aux.columns = ['customerid', 'qty_media', 'qty_total_comprada', 'unitprice_medio']
    
    # Merge de características complementarias
    df_rfm = pd.merge(df_rfm, df_aux, on='customerid')
    
    return df_rfm, fecha_referencia