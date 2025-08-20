import os
import re
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from github import Github, GithubException
import hashlib

# ------------------- Configuración GitHub -------------------
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = "yeifer125/cenada-predicciones"
GITHUB_BRANCH = "main"

if not GITHUB_TOKEN:
    print("⚠️ No hay token de GitHub configurado en GITHUB_TOKEN")
    repo = None
else:
    g = Github(GITHUB_TOKEN)
    try:
        repo = g.get_repo(REPO_NAME)
        print(f"✅ Conectado a GitHub: {repo.full_name}")
    except GithubException as e:
        print(f"❌ No se pudo acceder al repo: {e}")
        repo = None

# ------------------- Funciones GitHub -------------------
def hash_archivo(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def actualizar_github_si_cambia(local_path, remote_path, mensaje="Actualizar archivo"):
    if repo is None:
        print(f"⚠️ Repo no disponible. Se guarda solo localmente: {local_path}")
        return
    if not os.path.exists(local_path):
        print(f"⚠️ Archivo local no encontrado: {local_path}")
        return

    local_hash = hash_archivo(local_path)

    try:
        archivo = repo.get_contents(remote_path, ref=GITHUB_BRANCH)
        remoto_hash = hashlib.sha256(archivo.decoded_content).hexdigest()
        if local_hash == remoto_hash:
            print(f"ℹ️ No cambió: {remote_path}")
            return
        repo.update_file(
            path=archivo.path,
            message=mensaje,
            content=open(local_path, "rb").read(),
            sha=archivo.sha,
            branch=GITHUB_BRANCH
        )
        print(f"✅ Actualizado en GitHub: {remote_path}")
    except GithubException as e:
        if e.status == 404:
            repo.create_file(
                path=remote_path,
                message=mensaje,
                content=open(local_path, "rb").read(),
                branch=GITHUB_BRANCH
            )
            print(f"✅ Creado en GitHub: {remote_path}")
        else:
            print(f"❌ Error al subir {remote_path}: {e}")
    except Exception as e:
        print(f"❌ Error inesperado al leer {local_path}: {e}")

# ------------------- Funciones auxiliares -------------------
def nombre_valido(nombre):
    return re.sub(r'[\\/:"*?<>| ]', '_', nombre)

# ------------------- Carpetas -------------------
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

# ------------------- Descargar y actualizar historial -------------------
historial_path = "data/historial.csv"
url = 'https://pima-scraper.onrender.com/precios'

try:
    response = requests.get(url)
    response.raise_for_status()
    datos = pd.DataFrame(response.json())
except Exception as e:
    print(f"❌ Error al descargar datos: {e}")
    datos = pd.DataFrame()

if datos.empty:
    print("⚠️ No se descargaron datos nuevos. Se usará el historial local existente.")

# Convertir fecha y promedio
datos['fecha'] = pd.to_datetime(datos['fecha'], errors='coerce') if not datos.empty else pd.Series(dtype='datetime64[ns]')
datos['promedio'] = pd.to_numeric(datos['promedio'], errors='coerce') if not datos.empty else pd.Series(dtype='float')
datos = datos.dropna(subset=['fecha','promedio','producto']).sort_values('fecha') if not datos.empty else pd.DataFrame()

# Cargar historial existente
if os.path.exists(historial_path):
    historial = pd.read_csv(historial_path)
    historial['fecha'] = pd.to_datetime(historial['fecha'], errors='coerce')
    historial = historial.dropna(subset=['fecha','promedio','producto'])
else:
    historial = pd.DataFrame(columns=['producto','fecha','promedio'])

# ------------------- Actualizar columna 'clave' y agregar nuevos datos -------------------
if not datos.empty:
    datos['clave'] = datos['producto'] + datos['fecha'].dt.strftime('%Y-%m-%d')
    historial['clave'] = historial['producto'] + historial['fecha'].dt.strftime('%Y-%m-%d') if not historial.empty else []
    nuevos = datos[~datos['clave'].isin(historial['clave'])].drop(columns=['clave'])

    if not nuevos.empty:
        historial = pd.concat([historial, nuevos], ignore_index=True)
        historial.sort_values(['producto','fecha'], inplace=True)
        print(f"✅ Historial actualizado: {len(nuevos)} nuevos registros agregados")
    else:
        print("No hay datos nuevos que agregar")
else:
    print("Usando historial existente sin cambios.")

# ------------------- Guardar historial en formato YYYY-MM-DD -------------------
historial['fecha'] = historial['fecha'].dt.strftime('%Y-%m-%d')
historial['clave'] = historial['producto'] + historial['fecha']
historial.to_csv(historial_path, index=False)
actualizar_github_si_cambia(historial_path, "data/historial.csv", "Actualizar historial con nuevas fechas")

# ------------------- Generar predicciones -------------------
productos = historial['producto'].unique()
scaler = MinMaxScaler()

for producto in productos:
    try:
        print(f"\n🔹 Generando predicciones para: {producto}")
        df_prod = historial[historial['producto']==producto].copy().sort_values('fecha')
        df_prod['fecha'] = pd.to_datetime(df_prod['fecha'])
        df_prod.set_index('fecha', inplace=True)
        producto_safe = nombre_valido(producto)

        # ---------- LSTM ----------
        if len(df_prod) >= 6:
            precios_scaled = scaler.fit_transform(df_prod['promedio'].values.reshape(-1,1))
            X, y = [], []
            window_size = 5
            for i in range(window_size, len(precios_scaled)):
                X.append(precios_scaled[i-window_size:i,0])
                y.append(precios_scaled[i,0])
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1],1))
            model_path = f"models/lstm_{producto_safe}.keras"
            if os.path.exists(model_path):
                model_lstm = load_model(model_path)
                model_lstm.fit(X, y, epochs=5, batch_size=1, verbose=0)
            else:
                model_lstm = Sequential()
                model_lstm.add(LSTM(50, input_shape=(X.shape[1],1)))
                model_lstm.add(Dense(1))
                model_lstm.compile(optimizer='adam', loss='mean_squared_error')
                model_lstm.fit(X, y, epochs=50, batch_size=1, verbose=0)
            model_lstm.save(model_path)
            actualizar_github_si_cambia(model_path, f"models/lstm_{producto_safe}.keras", f"Actualizar LSTM {producto}")

            lstm_input = precios_scaled[-window_size:].reshape(1, window_size,1)
            preds = []
            for _ in range(10):
                pred = model_lstm.predict(lstm_input, verbose=0)
                preds.append(pred[0,0])
                lstm_input = np.append(lstm_input[:,1:,:], [[pred[0]]], axis=1)
            df_pred_lstm = pd.DataFrame({'pred': scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()})
            lstm_csv = f"predictions/predictions_lstm_{producto_safe}.csv"
            df_pred_lstm.to_csv(lstm_csv, index=False)
            actualizar_github_si_cambia(lstm_csv, lstm_csv, f"Actualizar predicciones LSTM {producto}")
        else:
            print(f"⚠️ No hay suficientes datos para LSTM: {producto}")

        # ---------- Prophet ----------
        if len(df_prod) >= 2:
            df_prophet = df_prod.reset_index().rename(columns={'fecha':'ds','promedio':'y'})
            model_prophet = Prophet(daily_seasonality=True)
            model_prophet.fit(df_prophet)
            future = model_prophet.make_future_dataframe(periods=10)
            forecast = model_prophet.predict(future)
            prophet_csv = f"predictions/predictions_prophet_{producto_safe}.csv"
            pd.DataFrame({'pred': forecast['yhat'].tail(10).values}).to_csv(prophet_csv, index=False)
            actualizar_github_si_cambia(prophet_csv, prophet_csv, f"Actualizar predicciones Prophet {producto}")
        else:
            print(f"⚠️ No hay suficientes datos para Prophet: {producto}")

        # ---------- ARIMA robusto ----------
        if len(df_prod) >= 6:
            try:
                df_prod_daily = df_prod.asfreq('D')  # Forzar frecuencia diaria
                arima_model = ARIMA(df_prod_daily['promedio'], order=(5,1,0))
                arima_fit = arima_model.fit()
                forecast_arima = arima_fit.forecast(steps=10)
            except Exception as e:
                print(f"⚠️ ARIMA estándar falló para {producto}: {e}")
                try:
                    arima_model = ARIMA(df_prod_daily['promedio'], order=(1,1,0))
                    arima_fit = arima_model.fit()
                    forecast_arima = arima_fit.forecast(steps=10)
                    print(f"✅ ARIMA fallback (1,1,0) aplicado para {producto}")
                except Exception as e2:
                    print(f"❌ ARIMA fallback también falló para {producto}: {e2}")
                    forecast_arima = None

            if forecast_arima is not None:
                arima_csv = f"predictions/predictions_arima_{producto_safe}.csv"
                pd.DataFrame({'pred': forecast_arima}).to_csv(arima_csv, index=False)
                actualizar_github_si_cambia(arima_csv, arima_csv, f"Actualizar predicciones ARIMA {producto}")
        else:
            print(f"⚠️ No hay suficientes datos para ARIMA: {producto}")

    except Exception as e:
        print(f"❌ Error general en producto {producto}: {e}")
