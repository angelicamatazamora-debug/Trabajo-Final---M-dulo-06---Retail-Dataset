import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataQualityGateError(Exception):
    """Excepción personalizada para fallas en las compuertas de calidad de datos."""
    pass

def ejecutar_data_quality_gates(df: pd.DataFrame) -> bool:
    """
    Ejecuta una suite estructurada de validaciones automáticas antes del modelado
    y muestra un reporte detallado de cada regla evaluada.
    """
    logger.info("="*60)
    logger.info("🔒 Ejecutando Data Quality Gates pre-modelado...")
    logger.info("="*60)
    
    # Diccionario para almacenar el estatus y detalle de cada compuerta
    reporte_gates = []
    
    try:
        # Regla 1: Volumen mínimo de filas
        minimum_rows = 100_000
        n_rows = df.shape[0]
        pass_rows = n_rows > minimum_rows
        reporte_gates.append({
            "Regla": "Volumen mínimo de registros", 
            "Esperado": f"> {minimum_rows:,}", 
            "Obtenido": f"{n_rows:,}", 
            "Estado": "✅ PASÓ" if pass_rows else "❌ FALLÓ"
        })
        assert pass_rows, f"Volumen insuficiente: el dataset tiene {n_rows} filas."

        # Regla 2: Umbral de duplicados
        max_dup_threshold = 0.01
        dup_ratio = df.duplicated().mean()
        pass_dup = dup_ratio < max_dup_threshold
        reporte_gates.append({
            "Regla": "Porcentaje de duplicados", 
            "Esperado": f"< {max_dup_threshold*100}%", 
            "Obtenido": f"{dup_ratio*100:.2f}%", 
            "Estado": "✅ PASÓ" if pass_dup else "❌ FALLÓ"
        })
        assert pass_dup, f"Demasiados duplicados: {dup_ratio:.2%}"

        # Regla 3: Nulos en columnas críticas
        columnas_criticas = ['customerid', 'invoiceno', 'quantity', 'unitprice']
        nulos_totales = sum(df[col].isna().sum() for col in columnas_criticas)
        pass_nulos = nulos_totales == 0
        reporte_gates.append({
            "Regla": "Valores nulos en columnas críticas", 
            "Esperado": "0 nulos", 
            "Obtenido": f"{nulos_totales} nulos", 
            "Estado": "✅ PASÓ" if pass_nulos else "❌ FALLÓ"
        })
        assert pass_nulos, f"Se encontraron {nulos_totales} nulos en columnas críticas."

        # Regla 4: Rangos numéricos válidos (> 0)
        min_qty = df['quantity'].min()
        min_price = df['unitprice'].min()
        pass_ranges = (min_qty > 0) and (min_price > 0)
        reporte_gates.append({
            "Regla": "Rangos lógicos (Quantity/UnitPrice > 0)", 
            "Esperado": "Mínimos > 0", 
            "Obtenido": f"Qty min: {min_qty}, Price min: {min_price}", 
            "Estado": "✅ PASÓ" if pass_ranges else "❌ FALLÓ"
        })
        assert (df['quantity'] > 0).all(), "Se encontraron quantities <= 0."
        assert (df['unitprice'] > 0).all(), "Se encontraron unitprices <= 0."

        # Regla 5: Tipos de datos correctos
        is_id_int = pd.api.types.is_integer_dtype(df['customerid'])
        is_date_dt = pd.api.types.is_datetime64_any_dtype(df['invoicedate'])
        pass_types = is_id_int and is_date_dt
        reporte_gates.append({
            "Regla": "Tipos de datos estrictos", 
            "Esperado": "customerid=int, invoicedate=datetime", 
            "Obtenido": f"customerid={df['customerid'].dtype}, date={df['invoicedate'].dtype}", 
            "Estado": "✅ PASÓ" if pass_types else "❌ FALLÓ"
        })
        assert is_id_int, "customerid debe ser entero."
        assert is_date_dt, "invoicedate debe ser datetime."

        # Imprimir tabla resumen limpia y estructurada en el notebook
        df_reporte = pd.DataFrame(reporte_gates)
        print("\n📋 REPORTE DE EJECUCIÓN DE DATA QUALITY GATES:")
        print(df_reporte)
        print("\n")

        logger.info("✅ ¡Todas las reglas de Data Quality Gates fueron superadas con éxito!")
        return True

    except AssertionError as e:
        df_reporte = pd.DataFrame(reporte_gates)
        print("\n📋 REPORTE DE FALLA EN DATA QUALITY GATES:")
        print(df_reporte)
        print("\n")
        logger.error(f"❌ FALLA EN DATA QUALITY GATE: {str(e)}")
        raise DataQualityGateError(f"El pipeline se ha detenido: {str(e)}")