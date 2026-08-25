# Importar Librerías
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
import sys 

# Configuración Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Diagnóstico de Calidad Ampliado
def diagnosticar_calidad(df):
    logger.info("="*60)
    logger.info("📊 DIAGNÓSTICO PROFUNDO DE CALIDAD DE DATOS")
    logger.info("="*60)
    
    logger.info(f"📌 Registros totales: {len(df):,}")
    logger.info(f"📌 Columnas totales: {len(df.columns):,}")
    
    # Valores Nulos
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df) * 100).round(2)
    
    tabla_nulos = pd.DataFrame({
        'Valores nulos': nulos,
        'Porcentaje (%)': nulos_pct
    }).sort_values('Valores nulos', ascending=False)
    
    logger.info("\nPrincipales columnas que presentan valores nulos:")
    print(tabla_nulos.head(10))
    print("\n")
    
    # Duplicados
    duplicados = df.duplicated().sum()
    duplicados_pct = (duplicados / len(df) * 100).round(2)
    logger.info(f"Registros duplicados exactos: {duplicados:,} ({duplicados_pct}%)")
    
    # Anomalías de negocio adaptadas a minúsculas normalizadas
    if 'quantity' in df.columns:
        cant_invalidas = (df['quantity'] <= 0).sum()
        logger.info(f"Registros con quantity <= 0 (devoluciones/anomalías): {cant_invalidas:,}")
        
    if 'unitprice' in df.columns:
        precio_invalidos = (df['unitprice'] < 0).sum()
        logger.info(f"Registros con unitprice < 0 (precios inválidos): {precio_invalidos:,}")

    return tabla_nulos

# Outliers (Valores Atípicos)
def detectar_outliers(df, columnas):
    resultados = []
    for columna in columnas:
        if columna not in df.columns:
            continue
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
        n_outliers = len(outliers)
        pct_outliers = round((n_outliers / len(df) * 100), 2)
        
        resultados.append({
            'Variable': columna,
            'Outliers': n_outliers,
            'Porcentaje (%)': pct_outliers,
            'Mínimo': df[columna].min(),
            'Máximo': df[columna].max()
        })
    return pd.DataFrame(resultados)

# Análisis de Correlaciones
def analizar_correlaciones(df, columnas_numericas):
    columnas_existentes = [col for col in columnas_numericas if col in df.columns]
    if len(columnas_existentes) < 2:
        return None
    return df[columnas_existentes].corr()

# Validación de Calidad (Data Quality Gates - Acorde a nombres normalizados)
def validate_data(df):
    logger.info("="*60)
    logger.info("Validación de Calidad en Proceso (Data Gates)...")
    logger.info("="*60)
    
    assert len(df) > 0, "El dataset está vacío"
    
    # Columnas obligatorias normalizadas
    required_cols = ['invoiceno', 'stockcode', 'quantity', 'invoicedate', 'unitprice', 'customerid']
    for col in required_cols:
        assert col in df.columns, f"Columna obligatoria faltante: {col}"
    
    # Reglas post-limpieza
    assert (df['quantity'] > 0).all(), "Se encontraron cantidades menores o iguales a cero"
    assert (df['unitprice'] > 0).all(), "Se encontraron precios menores o iguales a cero"
    
    logger.info("¡Todas las validaciones de calidad fueron aprobadas con éxito!")
    return True

# Limpieza de Datos
def clean_data(df):
    logger.info("="*60)
    logger.info("Iniciando Limpieza y Transformación de Datos")
    logger.info("="*60)
    
    df_clean = df.copy()
    registros_iniciales = len(df_clean)
    
    # Asumimos que el DataFrame ya viene con los nombres normalizados desde el notebook,
    # pero aseguramos consistencia por si se ejecuta de forma independiente.
    df_clean.columns = [col.strip().lower().replace(" ", "_") for col in df_clean.columns]
    
    # 1. Tipificación segura de fechas
    if 'invoicedate' in df_clean.columns:
        df_clean['invoicedate'] = pd.to_datetime(df_clean['invoicedate'], errors='coerce')
        logger.info("invoicedate convertido a datetime")

    # 2. Eliminar nulos en customerid
    if 'customerid' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=['customerid'])
        logger.info(f"Eliminados {before - len(df_clean):,} registros sin customerid")

    # 3. Filtrar cantidades y precios válidos (> 0)
    if 'quantity' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean['quantity'] > 0]
        logger.info(f"Eliminados {before - len(df_clean):,} registros con quantity <= 0")
        
    if 'unitprice' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean['unitprice'] > 0]
        logger.info(f"Eliminados {before - len(df_clean):,} registros con unitprice <= 0")

    # 4. Eliminar duplicados exactos
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    logger.info(f"Eliminados {before - len(df_clean):,} registros duplicados")

    # 5. Ajustar tipo final de customerid a entero limpio
    df_clean['customerid'] = df_clean['customerid'].astype('int64')

    registros_finales = len(df_clean)
    logger.info(f"Registros finales limpios: {registros_finales:,} (Conservados: {(registros_finales/registros_iniciales)*100:.2f}%)")
    
    return df_clean

# Pipeline Principal
def run_data_quality_pipeline(df):
    diagnosticar_calidad(df)
    detectar_outliers(df, ['quantity', 'unitprice'])
    analizar_correlaciones(df, ['quantity', 'unitprice'])
    df_clean = clean_data(df)
    validate_data(df_clean)
    
    # Guardar resultado
    processed_path = Path("data/processed/")
    processed_path.mkdir(parents=True, exist_ok=True)
    file_path = processed_path / "online_retail_clean.csv"
    df_clean.to_csv(file_path, index=False)
    logger.info(f"Datos limpios guardados exitosamente en: {file_path}")
    
    return df_clean