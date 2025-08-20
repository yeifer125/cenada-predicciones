import os
from github import Github, GithubException

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

# ------------------- Función para subir o actualizar archivos -------------------
def actualizar_github(local_path, remote_path, mensaje="Actualizar archivo"):
    if repo is None:
        print(f"⚠️ Repo no disponible. Se guarda solo localmente: {local_path}")
        return

    if not os.path.exists(local_path):
        print(f"⚠️ Archivo local no encontrado: {local_path}")
        return

    try:
        with open(local_path, "rb") as f:
            contenido = f.read()

        try:
            archivo = repo.get_contents(remote_path, ref=GITHUB_BRANCH)
            repo.update_file(
                path=archivo.path,
                message=mensaje,
                content=contenido,
                sha=archivo.sha,
                branch=GITHUB_BRANCH
            )
            print(f"✅ Actualizado en GitHub: {remote_path}")
        except GithubException as e:
            if e.status == 404:
                repo.create_file(
                    path=remote_path,
                    message=mensaje,
                    content=contenido,
                    branch=GITHUB_BRANCH
                )
                print(f"✅ Creado en GitHub: {remote_path}")
            else:
                print(f"❌ Error al subir {remote_path}: {e}")
    except Exception as e:
        print(f"❌ Error inesperado al leer {local_path}: {e}")

# ------------------- Subir todo el contenido de varias carpetas -------------------
carpetas = ["data", "models", "predictions"]

for carpeta in carpetas:
    if os.path.exists(carpeta):
        for root, _, files in os.walk(carpeta):
            for file in files:
                local_path = os.path.join(root, file)
                remote_path = local_path.replace("\\", "/")  # Para GitHub usar '/'
                actualizar_github(local_path, remote_path, f"Actualizar {remote_path}")
    else:
        print(f"⚠️ Carpeta no encontrada: {carpeta}")
