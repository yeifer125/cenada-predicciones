import os
import re
import warnings
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from github import Github
from io import StringIO
import subprocess
import tempfile
import time
import threading

# ---------------- Configuración página ----------------
st.set_page_config(page_title="by: cArbonAto", layout="wide")
st.title("📊 Posibles precios a futuro para Cenada")

# ---------------- Conexión a GitHub ----------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = "yeifer125/cenada-predicciones"
GITHUB_BRANCH = "main"

try:
    gh = Github(GITHUB_TOKEN)
    repo = gh.get_repo(REPO_NAME)
    print(f"✅ Conexión exitosa a GitHub: {REPO_NAME}")
except Exception as e:
    st.error("❌ No se pudo conectar a GitHub. Revisa el token o la red.")
    st.stop()

# ---------------- Funciones auxiliares ----------------
def nombre_valido(nombre):
    return re.sub(r'[\\/:"*?<>| ]', '_', nombre.strip())

@st.cache_data(ttl=72000)  # 20 horas
def leer_csv(path):
    try:
        archivo = repo.get_contents(path, ref=GITHUB_BRANCH)
        contenido = archivo.decoded_content.decode("utf-8")
        return pd.read_csv(StringIO(contenido))
    except Exception as e:
        return None

@st.cache_data(ttl=72000)
def leer_pred(producto, modelo):
    nombre_archivo = f"predictions/predictions_{modelo}_{nombre_valido(producto)}.csv"
    df = leer_csv(nombre_archivo)
    if df is None or "pred" not in df.columns:
        return None
    return df["pred"].values

# ---------------- Leer historial ----------------
historial = leer_csv("data/historial.csv")
if historial is None or historial.empty:
    st.error("No se encontró historial ni local ni en GitHub.")
    st.stop()

# ---------------- Preparar historial ----------------
historial.columns = [c.lower() for c in historial.columns]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    historial["fecha"] = pd.to_datetime(historial["fecha"], errors="coerce", dayfirst=True)
historial["promedio"] = pd.to_numeric(historial["promedio"], errors="coerce")
historial = historial.dropna(subset=["fecha", "producto", "promedio"])

# ---------------- Selector de producto ----------------
productos = historial["producto"].unique()
producto = st.selectbox("Selecciona un producto", productos)
df_producto = historial[historial["producto"] == producto].copy()
df_producto.set_index("fecha", inplace=True)
df_producto = df_producto.sort_index()

# ---------------- Leer predicciones ----------------
predicciones = {}
modelos = ["lstm", "prophet", "arima"]
faltantes = []
for modelo in modelos:
    pred = leer_pred(producto, modelo)
    predicciones[modelo] = pred
    if pred is None:
        faltantes.append(modelo)

if faltantes:
    st.warning(f"⚠️ Faltan predicciones para {producto}: {', '.join(faltantes)}")
    if all(predicciones[m] is None for m in modelos):
        st.stop()

# ---------------- Graficar histórico + predicciones con fechas reales ----------------
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12,6))

# Calcular fechas futuras
n_pred = max([len(p) for p in predicciones.values() if p is not None] + [0])
fechas_pred = pd.date_range(df_producto.index[-1] + pd.Timedelta(days=1), periods=n_pred)
fechas_totales = list(df_producto.index) + list(fechas_pred)

# Plot histórico con fechas reales
ax.plot(df_producto.index, df_producto["promedio"], label="Histórico", color="#00ffcc")

# Plot predicciones con fechas consecutivas al histórico
colores = {"lstm": "#66ff33", "prophet": "#ffa500", "arima": "#ff33cc"}
for modelo in modelos:
    pred = predicciones[modelo]
    if pred is not None:
        ax.plot(fechas_pred[:len(pred)], pred, "--o", label=modelo.upper(), color=colores[modelo])

# Ajustar ticks en eje X
tick_step = max(len(fechas_totales) // 10, 1)
ax.set_xticks(fechas_totales[::tick_step])
ax.set_xticklabels([f.strftime("%d/%m/%Y") for f in fechas_totales[::tick_step]], rotation=45, ha="right")

# Estilo
ax.set_title(f"Predicciones de {producto}", fontsize=16, color="white")
ax.set_xlabel("Fecha", color="white")
ax.set_ylabel("Precio", color="white")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# ---------------- Tabla ----------------
tabla_dict = {
    "Fecha": fechas_totales,
    "Histórico": list(df_producto["promedio"]) + [None]*n_pred
}
for modelo in modelos:
    pred = predicciones[modelo]
    tabla_dict[modelo.upper()] = [None]*len(df_producto) + (list(pred) if pred is not None else [None]*n_pred)

tabla = pd.DataFrame(tabla_dict)
st.dataframe(tabla, use_container_width=True)

# ---------------- Botón de actualización ----------------
st.markdown("---")
st.subheader("🔄 Actualizar datos y predicciones desde GitHub")

if st.button("Actualizar ahora"):
    with st.spinner("Actualizando..."):
        try:
            archivo_remote = repo.get_contents("updategit.py", ref=GITHUB_BRANCH)
            codigo = archivo_remote.decoded_content.decode("utf-8")
            
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as tmpfile:
                tmpfile.write(codigo)
                tmpfile_path = tmpfile.name

            resultado = subprocess.run(
                ["python3", tmpfile_path],
                capture_output=True,
                text=True,
                check=False
            )

            st.text("📄 Salida del script:")
            st.text(resultado.stdout)
            if resultado.stderr:
                st.error("⚠️ Errores durante la ejecución:")
                st.text(resultado.stderr)
        except Exception as e:
            st.error(f"❌ Error inesperado al actualizar: {e}")

# ---------------- Refrescar cada 20 horas ----------------
def refrescar_automatica():
    while True:
        time.sleep(72000)  # 20 horas
        st.experimental_rerun()

threading.Thread(target=refrescar_automatica, daemon=True).start()
