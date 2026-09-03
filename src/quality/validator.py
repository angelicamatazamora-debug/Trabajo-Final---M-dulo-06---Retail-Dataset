import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataQualityLogger")

def validate_data_quality(df: pd.DataFrame) -> dict:
    report = {
        "status": "PASS",
        "errors": [],
        "warnings": []
    }
    
    required_columns = ['customerid', 'invoiceno', 'stockcode', 'quantity', 'unitprice', 'invoicedate']
    
    # 1. Validación de Esquema (No hace return anticipado para evaluar lo demás)
    missing_cols = [col for col in required_columns if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in required_columns]
    if missing_cols:
        report["status"] = "FAIL"
        err_msg = f"Modificación de esquema - Columnas faltantes: {missing_cols}"
        report["errors"].append(err_msg)
        logger.error(err_msg)
    if extra_cols:
        warn_msg = f"Modificación de esquema - Columnas inesperadas: {extra_cols}"
        report["warnings"].append(warn_msg)
        logger.warning(warn_msg)

    # 2. Validación de Tipos de Datos (Solo si la columna existe)
    numeric_cols = ['quantity', 'unitprice']
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            report["status"] = "FAIL"
            err_msg = f"Tipo de dato incorrecto en '{col}': se esperaba numérico."
            report["errors"].append(err_msg)
            logger.error(err_msg)

    # 3. Validación de Duplicados
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        warn_msg = f"Se detectaron {dup_count} filas duplicadas."
        report["warnings"].append(warn_msg)
        logger.warning(warn_msg)

    # 4. Validación de Valores Nulos (Solo en columnas existentes)
    existing_required = [col for col in required_columns if col in df.columns]
    null_counts = df[existing_required].isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            report["status"] = "FAIL"
            err_msg = f"Valores faltantes (nulos) en '{col}': {count} registros."
            report["errors"].append(err_msg)
            logger.error(err_msg)

    # 5. Validación de Categorías Desconocidas (Country)
    if 'country' in df.columns:
        valid_countries = ['United Kingdom', 'France', 'Germany', 'EIRE', 'Spain', 'Belgium', 'Switzerland', 'Portugal']
        unknown_countries = df[~df['country'].isin(valid_countries)]['country'].unique()
        if len(unknown_countries) > 0:
            warn_msg = f"Categoría desconocida detectada en 'country': {list(unknown_countries)}"
            report["warnings"].append(warn_msg)
            logger.warning(warn_msg)

    # 6. Validación de Valores Atípicos Extremos (Solo si existen)
    if 'quantity' in df.columns:
        try:
            if (df['quantity'] > 50000).any() or (df['quantity'] <= 0).any():
                report["status"] = "FAIL"
                err_msg = "Outlier extremo o cantidad inválida detectada en 'quantity'."
                report["errors"].append(err_msg)
                logger.error(err_msg)
        except Exception:
            pass
            
    if 'unitprice' in df.columns:
        try:
            if (pd.to_numeric(df['unitprice'], errors='coerce') < 0).any():
                report["status"] = "FAIL"
                err_msg = "Precio unitario negativo detectado."
                report["errors"].append(err_msg)
                logger.error(err_msg)
        except Exception:
            pass

    if report["status"] == "FAIL":
        logger.error("Resultado de Calidad: BLOQUEADO (FAIL)")
    else:
        logger.info("Resultado de Calidad: APROBADO con advertencias")

    return report