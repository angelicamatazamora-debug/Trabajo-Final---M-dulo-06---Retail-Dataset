# test_retrain.py
from src.training.retrainer import retrain_kmeans_model

if __name__ == "__main__":
    # Simulamos el reentrenamiento apuntando al dataset limpio o a un batch de producción validado
    result = retrain_kmeans_model(new_data_path="data/processed/online_retail_clean.csv")
    print("\n--- REPORTE DE REENTRENAMIENTO ---")
    print(result)