# Importar Librerías
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys 
import re 

# Agregar la raíz del proyecto al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuración Loggin
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Diagnóstico de Calidad

def diagnosticar_calidad(df):
    """
    Realiza un diagnóstico completo de la calidad de los datos.
    
    Args:
        df: DataFrame a diagnosticar
    
    Returns:
        DataFrame: Tabla con análisis de valores nulos
    """
    logger.info("="*60)
    logger.info("📊 DIAGNÓSTICO DE CALIDAD DE DATOS")
    logger.info("="*60)
    
    # Resumen del Dataset
    logger.info(f"📌 Registros totales: {len(df):,}")
    logger.info(f"📌 Columnas totales: {len(df.columns):,}")
    
    # Valores Nulos
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df) * 100).round(2)
    
    tabla_nulos = pd.DataFrame({
        'Valores nulos': nulos,
        'Porcentaje (%)': nulos_pct
    }).sort_values('Valores nulos', ascending=False)
    
    logger.info("\n Principales columnas que presentan valores nulos:")
    print(tabla_nulos.head(10))
    print("\n")
    # Vacíos Columnas
    columnas_vacias = tabla_nulos[tabla_nulos['Porcentaje (%)'] > 95].index.tolist()
    if columnas_vacias:
        logger.info(f"Columnas con > 95% nulos")
        for col in columnas_vacias:
            logger.info(f"   - {col}")
    
    # Duplicados
    duplicados = df.duplicated().sum()
    duplicados_pct = (duplicados / len(df) * 100).round(2)
    logger.info(f"\n Registros duplicados: {duplicados:,} ({duplicados_pct}%)")
    
    # Tipos de Datos
    logger.info("\n Tipos de datos del dataset:")
    print(df.dtypes.value_counts())
    
    return tabla_nulos

#Outliers (Valores Atípicos)

def detectar_outliers(df, columnas):
    """
    Detecta valores atípicos usando el método IQR.
    
    Args:
        df: DataFrame a analizar
        columnas: Lista de columnas numéricas a revisar
    
    Returns:
        DataFrame: Tabla con resultados de outliers
    """
    logger.info("\n" + "="*60)
    logger.info("Identificación de Outliers")
    logger.info("="*60)
    logger.info("Método: IQR")
    
    resultados = []
    for columna in columnas:
        if columna not in df.columns:
            continue
        
        # Calcular cuartiles
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        
        # Límites para detectar outliers
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Contar outliers
        outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
        n_outliers = len(outliers)
        pct_outliers = round((n_outliers / len(df) * 100), 2)
        
        resultados.append({
            'Variable': columna,
            'Outliers': n_outliers,
            'Porcentaje (%)': pct_outliers,
            'Mínimo': df[columna].min(),
            'Máximo': df[columna].max(),
            'Q1 (25%)': Q1,
            'Mediana (50%)': df[columna].median(),
            'Q3 (75%)': Q3,
            'Límite inferior': limite_inferior,
            'Límite superior': limite_superior
        })
        
        if n_outliers > 0:
            logger.info(f"  Inferior al Limite  {columna}: {n_outliers:,} outliers ({pct_outliers}%)")
            logger.info(f"     Rango normal: [{limite_inferior:.2f}, {limite_superior:.2f}]")
        else:
            logger.info(f"   Suerior al Límite {columna}: {n_outliers} outliers")
    
    return pd.DataFrame(resultados)

#Correlaciones

def analizar_correlaciones(df, columnas_numericas):
    """
    Analiza las correlaciones entre variables numéricas.
    
    Args:
        df: DataFrame a analizar
        columnas_numericas: Lista de columnas numéricas
    
    Returns:
        DataFrame: Matriz de correlación
    """
    logger.info("\n" + "="*60)
    logger.info("Análisis de Correlaciones")
    logger.info("="*60)
    
    columnas_existentes = [col for col in columnas_numericas if col in df.columns]
    
    if len(columnas_existentes) < 2:
        logger.warning("No hay suficientes columnas numéricas para correlación")
        return None
    
    # Calcular matriz de correlación
    matriz_correlacion = df[columnas_existentes].corr()
    
    # Identificar correlaciones altas
    correlaciones_altas = []
    for i in range(len(matriz_correlacion.columns)):
        for j in range(i+1, len(matriz_correlacion.columns)):
            corr = matriz_correlacion.iloc[i, j]
            if abs(corr) > 0.7:
                correlaciones_altas.append({
                    'Variable 1': matriz_correlacion.columns[i],
                    'Variable 2': matriz_correlacion.columns[j],
                    'Correlación': corr
                })
    
    if correlaciones_altas:
        logger.info("Correlaciones fuertes (>0.7) encontradas:")
        for item in correlaciones_altas:
            logger.info(f"   {item['Variable 1']} ↔ {item['Variable 2']}: {item['Correlación']:.2f}")
        logger.info(" Nota: Variables con alta correlación pueden ser redundantes")
    else:
        logger.info("No se encontraron correlaciones fuertes (>0.7)")
    
    return matriz_correlacion

# Validación de Calidad
def validate_data(df):
    """
    Valida la calidad de los datos con 5 reglas automáticas.
    
    Estas son las "Data Quality Gates" que pide el proyecto:
    - Cada regla debe pasar para que el pipeline continúe
    - Si alguna falla, el proceso se detiene
    
    Args:
        df: DataFrame a validar
    
    Returns:
        bool: True si pasa todas las validaciones
    
    Raises:
        AssertionError: Si alguna validación falla
    """
    logger.info("="*60)
    logger.info("Validación de Calidad en Proceso...")
    logger.info("="*60)
    
    # REGLA 1: Dataset no vacío
    logger.info("Primera Regla: Dataset no vacío")
    assert len(df) > 0, "El dataset está vacío"
    logger.info(f"Dataset contiene {len(df):,} registros")

    # REGLA 2: Columnas obligatorias
    logger.info("Segunda Regla: Columnas obligatorias")
    required_cols = ['InvoiceNo', 'StockCode', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID']
    missing_cols = [col for col in required_cols if col not in df.columns]
    assert len(missing_cols) == 0, f"Columnas faltantes: {missing_cols}"
    logger.info(f"Todas las columnas obligatorias están presentes")
    
    # REGLA 3: Sin valores nulos en columnas críticas
    logger.info("Tercera regla: Valores nulos en columnas críticas")
    critical_cols = ['InvoiceNo', 'Quantity', 'UnitPrice']
    for col in critical_cols:
        null_count = df[col].isnull().sum()
        assert null_count == 0, f" {col} tiene {null_count} valores nulos"
        logger.info(f"{col}: {null_count} valores nulos")
    
    # REGLA 4: Cantidades y precios positivos
    logger.info("Cuarta Regla: Cantidades y precios positivos")
    assert (df['Quantity'] > 0).all(), "Cantidades negativas o cero encontradas"
    assert (df['UnitPrice'] > 0).all(), "Precios negativos o cero encontrados"
    logger.info("Cantidades y precios positivos")

    # REGLA 5: Menos de 5% de duplicados
    logger.info("Quinta Regla: Registros duplicados")
    dup_count = df.duplicated().sum()
    dup_pct = dup_count / len(df) * 100
    assert dup_pct < 5, f"{dup_pct:.2f}% de duplicados (máximo 5%)"
    logger.info(f"{dup_pct:.2f}% de duplicados (<5%)")
    
    logger.info("="*60)
    logger.info("Todas las Validaciones Aprobadas")
    logger.info("="*60)
    return True

# Limpieza de Datos
def clean_data(df):
    """
    Limpia los datos eliminando registros problemáticos.
    
    Args:
        df: DataFrame a limpiar
    
    Returns:
        DataFrame: Datos limpios
    """
    logger.info("="*60)
    logger.info("Iniciando Limpieza de Datos")
    logger.info("="*60)
    
    df_clean = df.copy()
    registros_iniciales = len(df_clean)
    logger.info(f"Registros iniciales: {registros_iniciales:,}")
    
    # Eliminar filas con CustomerID nulo. No se puede segmentar un cliente sin identificarlo
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=['CustomerID'])
    after = len(df_clean)
    eliminados = before - after
    logger.info(f" Eliminados {eliminados:,} registros sin CustomerID")

 # Eliminar transacciones con cantidad <= 0
    before = len(df_clean)
    df_clean = df_clean[df_clean['Quantity'] > 0]
    after = len(df_clean)
    eliminados = before - after
    if eliminados > 0:
        logger.info(f" Eliminados {eliminados:,} registros con cantidad <= 0")
    
    # Eliminar transacciones con precio <= 0
    before = len(df_clean)
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    after = len(df_clean)
    eliminados = before - after
    if eliminados > 0:
        logger.info(f" Eliminados {eliminados:,} registros con precio <= 0")
    
    # Eliminar duplicados exactos
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)
    eliminados = before - after
    if eliminados > 0:
        logger.info(f"Eliminados {eliminados:,} registros duplicados")

    # Eliminar descripciones no válidas. Descripciones como 'Manual' no son productos reales
    before = len(df_clean)
    invalid_desc = ['Manual', 'Adjust', 'test', 'Discount', 'Sample']
    invalid_desc_escapado = [re.escape(desc) for desc in invalid_desc]
    df_clean = df_clean[~df_clean['Description'].str.contains(
        '|'.join(invalid_desc_escapado), case=False, na=False, regex=True
    )]
    after = len(df_clean)
    eliminados = before - after
    if eliminados > 0:
        logger.info(f"  Eliminados {eliminados:,} registros con descripciones no válidas")

     #  Convertir tipos de datos
    df_clean['CustomerID'] = df_clean['CustomerID'].astype('int64')
    logger.info("CustomerID convertido a int")
    
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    logger.info(" InvoiceDate convertido a datetime")
    
    registros_finales = len(df_clean)
    total_eliminados = registros_iniciales - registros_finales
    pct_eliminados = (total_eliminados / registros_iniciales) * 100
    
    logger.info("="*60)
    logger.info("RESUMEN DE LIMPIEZA")
    logger.info("="*60)
    logger.info(f"   Registros iniciales: {registros_iniciales:,}")
    logger.info(f"   Registros finales: {registros_finales:,}")
    logger.info(f"   Registros eliminados: {total_eliminados:,} ({pct_eliminados:.2f}%)")
    logger.info("="*60)
    
    return df_clean

#Pipeline de Calidad
def run_data_quality_pipeline(df):
    """
    Ejecuta todo el pipeline de calidad de datos:
    1. Diagnóstico (nulos, duplicados, tipos)
    2. Detección de outliers (valores atípicos)
    3. LIMPIEZA DE DATOS (antes de validar)
    4. Validación (5 reglas automáticas)
    5. Guardado de datos limpios
    """
    logger.info("="*60)
    logger.info("Iniciando Pipeline de Calidad de Datos")
    logger.info("="*60)
    
    # Diagnóstico de calidad
    diagnosticar_calidad(df)
    
    # Detectar outliers 
    columnas_outliers = ['Quantity', 'UnitPrice']
    detectar_outliers(df, columnas_outliers)
    
    # Análisis de correlaciones
    columnas_numericas = ['Quantity', 'UnitPrice']
    analizar_correlaciones(df, columnas_numericas)
    
    # Limpieza de datos
    df_clean = clean_data(df)

     # Validación 
    validate_data(df_clean)
    
    # Guardar datos limpios
    processed_path = Path("data/processed/")
    processed_path.mkdir(parents=True, exist_ok=True)
    
    file_path = processed_path / "online_retail_clean.csv"
    df_clean.to_csv(file_path, index=False)
    logger.info(f"Datos limpios guardados en: {file_path}")
    
    logger.info("="*60)
    logger.info("Pipeline de Calidad Completo")
    logger.info("="*60)
    
    return df_clean

# Ejecución 
if __name__ == "__main__":
    from ingestion.Ingesta import ingest_data
    
    print("\n" + "="*60)
    print("Ejecutando Pipeline de Calidad de Datos")
    print("="*60)
    
    df_raw = ingest_data()
    df_clean = run_data_quality_pipeline(df_raw)
    
    print("\n" + "="*60)
    print("Resumen -  Calidad de Datos")
    print("="*60)
    print(f"Registros originales: {len(df_raw):,}")
    print(f"Registros después de limpieza: {len(df_clean):,}")
    print(f"Registros eliminados: {len(df_raw) - len(df_clean):,}")
    print(f"Porcentaje de datos conservados: {(len(df_clean)/len(df_raw)*100):.2f}%")
    
    print(f"\n Columnas después de limpieza:")
    for col in df_clean.columns:
        print(f"   - {col} ({df_clean[col].dtype})")
    
    print(f"\n Valores nulos después de limpieza:")
    for col in df_clean.columns:
        null_count = df_clean[col].isnull().sum()
        if null_count > 0:
            print(f"    {col}: {null_count:,} nulos")
        else:
            print(f"    {col}: {null_count} nulos")
    
    print(f"\n Primeras 5 filas:")
    print(df_clean.head())
    print("="*60)
