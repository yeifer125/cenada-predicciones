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

# ---------------- Configuración página ----------------
st.set_page_config(page_title="Predicción Cenada", layout="wide")
st.title("📊 Predicciones instantáneas de precios")

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
    """Limpia nombre para coincidir con los archivos en GitHub."""
    return re.sub(r'[\\/:"*?<>| ]', '_', nombre.strip())

@st.cache_data(ttl=300)
def leer_csv(path):
    try:
        print(f"Intentando leer archivo: {path}")  # Para depuración
        archivo = repo.get_contents(path, ref=GITHUB_BRANCH)
        contenido = archivo.decoded_content.decode("utf-8")
        return pd.read_csv(StringIO(contenido))
    except Exception as e:
        return None  # No mostramos warning aquí para limpiar la consola

@st.cache_data(ttl=300)
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

# Silenciar warnings innecesarios de fechas
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

# Advertencia limpia si faltan predicciones
if faltantes:
    st.warning(f"⚠️ Faltan predicciones para {producto}: {', '.join(faltantes)}")
    if all(predicciones[m] is None for m in modelos):
        st.stop()

# ---------------- Crear fechas futuras ----------------
n_pred = max(len(p) for p in predicciones.values() if p is not None)
future_dates = pd.date_range(df_producto.index[-1] + pd.Timedelta(days=1), periods=n_pred)

# ---------------- Graficar ----------------
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df_producto.index, df_producto["promedio"], label="Histórico", color="#00ffcc")

colores = {"lstm": "#66ff33", "prophet": "#ffa500", "arima": "#ff33cc"}
for modelo in modelos:
    if predicciones[modelo] is not None:
        ax.plot(future_dates[:len(predicciones[modelo])],
                predicciones[modelo], "--o", label=modelo.upper(), color=colores[modelo])

ax.set_title(f"Predicciones de {producto}", fontsize=16, color="white")
ax.set_xlabel("Fecha", color="white")
ax.set_ylabel("Precio", color="white")
ax.grid(True, alpha=0.3)
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)

# ---------------- Tabla con histórico y predicciones ----------------
n_hist = min(10, len(df_producto))
hist_index = df_producto.index[-n_hist:]
hist_values = df_producto["promedio"].tail(n_hist)

tabla_dict = {"Fecha": list(hist_index) + list(future_dates), "Histórico": list(hist_values) + [None]*len(future_dates)}
for modelo in modelos:
    tabla_dict[modelo.upper()] = [None]*n_hist + (list(predicciones[modelo]) if predicciones[modelo] is not None else [None]*len(future_dates))

tabla = pd.DataFrame(tabla_dict)
st.dataframe(tabla, use_container_width=True)
