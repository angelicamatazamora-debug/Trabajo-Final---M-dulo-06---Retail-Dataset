# Importar librerías
import pandas as pd
from pathlib import Path
import logging
import requests
#Configuración del Login
logging.basicConfig(
    level=logging.INFO,  # Muestra mensajes informativos, warnings y errores
    format='%(asctime)s - %(levelname)s - %(message)s'  # Formato: [HORA] - [NIVEL] - [MENSAJE]
)
# Logger específico para este archivo
logger = logging.getLogger(__name__)

# Descarga del dataset
def download_online_retail():
    """
    Descarga el dataset Online Retail desde la fuente oficial de UCI.
    
    Returns:
        DataFrame de pandas con los datos del dataset
    
    Raises:
        Exception: Si no se puede descargar o cargar el dataset
    """
    try:
        # Fuente oficial: UCI Machine Learning Repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        # Ruta donde se guardará el archivo
        raw_path = Path("data/raw/")

        # mkdir(parents=True, exist_ok=True) crea la carpeta si no existe
        # parents=True: crea todas las carpetas intermedias necesarias
        # exist_ok=True: no da error si la carpeta ya existe
        raw_path.mkdir(parents=True, exist_ok=True)
        
        #Ruta completa del archivo
    
        file_path = raw_path / "Online Retail.xlsx"
        
        # Si ya está descargado, se carga directamente 
        if file_path.exists():
            logger.info("Dataset ya existe en disco. Cargando dataset...")
            return pd.read_excel(file_path) 
        
        # Si no existe, proceder con la descarga
        logger.info("Dataset no encontrado. Iniciando descarga...")
        
        # Petición HTTP para descargar el archivo
        # timeout=60: esperar máximo 60 segundos antes de dar tiempo de espera agotado
        response = requests.get(url, timeout=None)
        
        # Verificar que la descarga fue exitosa
        response.raise_for_status()
        
        # Guardar el archivo en disco
        with open(file_path, "wb") as f:
            f.write(response.content)  
        logger.info("Dataset descargado exitosamente")
        
        # Cargar el archivo descargado como DataFrame
        df = pd.read_excel(file_path)
        logger.info(f"Dataset cargado: {df.shape[0]:,} filas y {df.shape[1]} columnas")
        
        return df
        
    except requests.exceptions.Timeout:
        # Error específico: tiempo de espera agotado
        logger.error("Tiempo de espera agotado.")
        raise
        
    except requests.exceptions.RequestException as e:
        # Error específico: problema con la petición HTTP
        logger.error(f"Error al descargar el dataset: {e}")
        logger.info("Favor descargar manualmente desde:")
        logger.info("https://archive.ics.uci.edu/dataset/352/online+retail")
        logger.info("Y colocar el archivo en: data/raw/Online Retail.xlsx")
        raise
        
    except Exception as e:
        # Error genérico
        logger.error(f"Error inesperado: {e}")
        raise

# Ingesta de los Datos

def ingest_data():
    """
    Función principal que ejecuta el proceso de ingesta.
    
    Returns:
        DataFrame de pandas con los datos cargados
    
    Esta función es la que se llama desde otros scripts
    para obtener los datos.
    """
    logger.info("Iniciando proceso de ingesta de datos...")
    
    # Llamar a la función que descarga/carga el dataset
    df = download_online_retail()
    
    # Mostrar un resumen de los datos cargados
    logger.info(f"Ingesta completada: {df.shape[0]:,} registros cargados")
    
    return df

#Ejecución del Script

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PROCESO DE INGESTA - ONLINE RETAIL")
    print("="*60)
    
    # Ejecutar la ingesta
    df = ingest_data()
    
    # Mostrar información detallada del dataset
    print("\n" + "="*60)
    print("RESUMEN DEL DATASET")
    print("="*60)
    print(f"Total de registros: {len(df):,}")
    print(f"Total de columnas: {len(df.columns)}")
    print(f"\n Columnas disponibles:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col} ({df[col].dtype})")
    
    print(f"\n Primeras 5 filas:")
    print(df.head())
    
    print(f"\n Valores nulos por columna:")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"    {col}: {null_count:,} valores nulos ({null_count/len(df)*100:.2f}%)")
        else:
            print(f"    {col}: {null_count} valores nulos")
    
    print("="*60)
