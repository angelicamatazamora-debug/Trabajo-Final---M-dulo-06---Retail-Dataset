import pandas as pd
from pathlib import Path
from src.quality.validator import validate_data_quality

DATA_DIR = Path("data")
df_base = pd.read_csv(DATA_DIR / "processed" / "online_retail_clean.csv").sample(n=50)

# Simular batch de producción contaminado con múltiples defectos
df_contaminated = df_base.copy()

# 1. Missing values
df_contaminated.loc[df_contaminated.index[0], 'customerid'] = None

# 2. Duplicated rows
df_contaminated = pd.concat([df_contaminated, df_contaminated.iloc[[0]]], ignore_index=True)

# 3. Extreme outlier / Valor inválido
df_contaminated.loc[df_contaminated.index[1], 'quantity'] = -9999

# 4. Incorrect datatype (inyectando string en columna numérica)
df_contaminated['unitprice'] = df_contaminated['unitprice'].astype(object)
df_contaminated.loc[df_contaminated.index[2], 'unitprice'] = "treinta"

# 5. Unknown category
df_contaminated.loc[df_contaminated.index[3], 'country'] = "UNKNOWN_NEW_COUNTRY"

# 6. Schema modification (Eliminar una columna requerida)
df_contaminated = df_contaminated.drop(columns=['invoiceno'])

# Ejecutar validación
result = validate_data_quality(df_contaminated)
print("\n--- REPORTE FINAL DE CALIDAD ---")
print(result)