import numpy as np
import pandas as pd
from pathlib import Path
from src.monitoring.monitor import evaluate_data_drift

def run_production_simulation():
    DATA_DIR = Path("data")
    df_full = pd.read_csv(DATA_DIR / "processed" / "online_retail_clean.csv")
    
    # Transformar a formato RFM base (referencia)
    def to_rfm(df):
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
        return rfm.reset_index().dropna()

    df_ref = to_rfm(df_full)
    features = ["recency", "frequency", "monetary", "qty_media", "qty_total_comprada", "unitprice_medio"]

    # --- BATCH 1: Producción normal (Sin drift esperado) ---
    batch_1 = df_ref.sample(n=500, random_state=10)

    # --- BATCH 2: Drift moderado (Aumento leve en monetary y frequency) ---
    batch_2 = df_ref.sample(n=500, random_state=20).copy()
    batch_2['monetary'] = batch_2['monetary'] * 1.35
    batch_2['frequency'] = batch_2['frequency'] + 1

    # --- BATCH 3: Drift severo/crítico (Cambio drástico en recency y monetary) ---
    batch_3 = df_ref.sample(n=500, random_state=30).copy()
    batch_3['recency'] = batch_3['recency'] * 2.2
    batch_3['monetary'] = batch_3['monetary'] * 3.0

    batches = {"BATCH 1 (Normal)": batch_1, "BATCH 2 (Moderado)": batch_2, "BATCH 3 (Crítico)": batch_3}

    print("=== SIMULACIÓN DE LOTES DE PRODUCCIÓN Y DRIFT (PSI) ===")
    for name, batch in batches.items():  # <-- Agrega .items() aquí
        print(f"\n--- Evaluando {name} ---")
        report = evaluate_data_drift(df_ref, batch, features)
        for var, res in report.items():
            print(f"  Variable: {var:<20} | PSI: {res['psi']:.4f} | Estado: {res['status']}")

if __name__ == "__main__":
    run_production_simulation()