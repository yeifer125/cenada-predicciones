import pandas as pd

# Ruta al archivo original
input_csv = "data/historial.csv"
# Ruta al archivo corregido (NO sobrescribe el original)
output_csv = "data/historial_fechas_corregidas.csv"

# Cargar el CSV
df = pd.read_csv(input_csv)

# Asegurarnos que la columna 'fecha' es tipo datetime
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

# Transformar al formato YYYY-DD-MM
df['fecha'] = df['fecha'].dt.strftime('%Y-%d-%m')

# Guardar archivo corregido
df.to_csv(output_csv, index=False)

print(f"✅ Fechas transformadas y guardadas en: {output_csv}")
