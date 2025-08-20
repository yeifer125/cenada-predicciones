import streamlit as st
import pandas as pd
import os

st.title("Comparador de Historiales")

# Selección de archivos
archivo1 = st.file_uploader("Selecciona el historial principal", type="csv")
archivo2 = st.file_uploader("Selecciona el historial secundario", type="csv")

if archivo1 and archivo2:
    df_principal = pd.read_csv(archivo1)
    df_secundario = pd.read_csv(archivo2)

    st.subheader("Vista previa del historial principal")
    st.dataframe(df_principal.head())

    st.subheader("Vista previa del historial secundario")
    st.dataframe(df_secundario.head())

    if st.button("Comparar y generar historialcompe.csv"):
        # Crear columna 'clave' si no existe
        for df in [df_principal, df_secundario]:
            if 'clave' not in df.columns:
                df['clave'] = df['producto'] + pd.to_datetime(df['fecha']).dt.strftime('%Y-%m-%d')

        # Filtrar nuevos registros
        nuevos = df_secundario[~df_secundario['clave'].isin(df_principal['clave'])]
        historialcompe = pd.concat([df_principal, nuevos], ignore_index=True)
        historialcompe.to_csv("historialcompe.csv", index=False)
        st.success(f"Archivo 'historialcompe.csv' generado con {len(nuevos)} nuevos registros")
