# Importación de librerías y módulos
import logging
from pathlib import Path
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def ingest_data() -> pd.DataFrame:
    """
    Descarga el dataset Online Retail desde el repositorio oficial de la UCI 
    usando ucimlrepo y lo almacena localmente en formato CSV.
    """
    logger.info("Iniciando proceso de ingesta de datos...")
    
    try:
        raw_path = Path("data/raw")
        raw_path.mkdir(parents=True, exist_ok=True)
        
        file_path = raw_path / "online_retail.csv"
        
        # 1. Comprobación de idempotencia: Si ya existe, nos ahorramos la descarga
        if file_path.exists():
            logger.info(f"Dataset ya encontrado localmente en '{file_path}'. Cargando desde disco...")
            df = pd.read_csv(file_path)
            logger.info(f"Dataset cargado exitosamente: {df.shape[0]:,} filas y {df.shape[1]} columnas.")
            return df
        
        # 2. Si no existe, procedemos a descargarlo mediante la API oficial
        logger.info("Dataset no encontrado en disco. Descargando desde la API de UCI (ID: 352)...")
        online_retail = fetch_ucirepo(id=352)
        
        # 3. Extraer el DataFrame original
        df = online_retail.data.original
        
        # 4. Guardar los datos crudos en formato CSV estándar
        df.to_csv(file_path, index=False)
        logger.info(f"Dataset descargado y guardado exitosamente en: {file_path}")
        logger.info(f"Dimensiones del dataset raw: {df.shape[0]:,} filas, {df.shape[1]} columnas.")
        
        return df
        
    except Exception as e:
        logger.error(f"Error crítico durante el proceso de ingesta: {e}")
        raise

if __name__ == "__main__":
    df_resultado = ingest_data()
