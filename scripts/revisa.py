import os
from github import Github, GithubException
import hashlib
import re

# ------------------- Configuración GitHub -------------------
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = "yeifer125/cenada-predicciones"
GITHUB_BRANCH = "main"

if not GITHUB_TOKEN:
    raise ValueError("No hay token de GitHub configurado en GITHUB_TOKEN")

g = Github(GITHUB_TOKEN)

try:
    repo = g.get_repo(REPO_NAME)
    print(f"✅ Conectado a GitHub: {repo.full_name}")
except GithubException as e:
    print(f"❌ No se pudo acceder al repo: {e}")
    repo = None

# ------------------- Función para normalizar nombres -------------------
def nombre_valido(nombre):
    return re.sub(r'[\\/:"*?<>| ]', '_', nombre.strip())

# ------------------- Función para calcular hash de un archivo -------------------
def hash_archivo(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

# ------------------- Función para subir o actualizar archivos solo si cambian -------------------
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

# ------------------- Subir archivos de varias carpetas -------------------
carpetas = ["data", "models", "predictions"]

for carpeta in carpetas:
    if os.path.exists(carpeta):
        for root, _, files in os.walk(carpeta):
            for file in files:
                local_path = os.path.join(root, file)

                # Normalizar nombres solo si es carpeta predictions
                if carpeta == "predictions":
                    partes = file.split("_")
                    if len(partes) >= 3:  # Ej: predictions_lstm_Aguacate Hass.csv
                        prefijo = "_".join(partes[:2])
                        producto = "_".join(partes[2:])
                        producto_safe = nombre_valido(producto)
                        file = f"{prefijo}_{producto_safe}.csv"

                remote_path = os.path.join(root, file).replace("\\", "/")  # GitHub usa '/'
                actualizar_github_si_cambia(local_path, remote_path, f"Actualizar {remote_path}")
    else:
        print(f"⚠️ Carpeta no encontrada: {carpeta}")
