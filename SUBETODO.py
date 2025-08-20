import os
from github import Github, GithubException
import hashlib
import re

# ------------------- Configuración GitHub -------------------
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = "yeifer125/cenada-predicciones"
GITHUB_BRANCH = "main"

if not GITHUB_TOKEN:
    raise ValueError("❌ No hay token de GitHub configurado en GITHUB_TOKEN")

g = Github(GITHUB_TOKEN)
try:
    repo = g.get_repo(REPO_NAME)
    print(f"✅ Conectado a GitHub: {repo.full_name}")
except GithubException as e:
    print(f"❌ No se pudo acceder al repo: {e}")
    repo = None

# ------------------- Funciones -------------------
def nombre_valido(nombre):
    """Quita caracteres inválidos para GitHub"""
    return re.sub(r'[\\/:"*?<>| ]', '_', nombre.strip())

def hash_archivo(path):
    """Calcula hash sha256 de un archivo local"""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def actualizar_github_si_cambia(local_path, remote_path, mensaje="Actualizar archivo"):
    """Sube o actualiza un archivo en GitHub solo si cambió"""
    if repo is None:
        return "no_repo"

    if not os.path.exists(local_path):
        return "no_local"

    local_hash = hash_archivo(local_path)

    try:
        archivo = repo.get_contents(remote_path, ref=GITHUB_BRANCH)
        remoto_hash = hashlib.sha256(archivo.decoded_content).hexdigest()

        if local_hash == remoto_hash:
            return "sin_cambios"

        repo.update_file(
            path=archivo.path,
            message=mensaje,
            content=open(local_path, "rb").read(),
            sha=archivo.sha,
            branch=GITHUB_BRANCH
        )
        return "actualizado"

    except GithubException as e:
        if e.status == 404:
            repo.create_file(
                path=remote_path,
                message=mensaje,
                content=open(local_path, "rb").read(),
                branch=GITHUB_BRANCH
            )
            return "creado"
        else:
            print(f"❌ Error al subir {remote_path}: {e}")
            return "error"
    except Exception as e:
        print(f"❌ Error inesperado con {local_path}: {e}")
        return "error"

# ------------------- Subir proyecto completo -------------------
proyecto_root = os.path.abspath(".")
ignorar_carpetas = {".git", "__pycache__", ".venv", ".vs"}

resumen = {"creado": [], "actualizado": [], "sin_cambios": [], "errores": []}

for root, dirs, files in os.walk(proyecto_root):
    # Ignorar carpetas basura
    dirs[:] = [d for d in dirs if d not in ignorar_carpetas]

    for file in files:
        local_path = os.path.join(root, file)

        # Normalizar nombre para GitHub
        nombre_file = nombre_valido(file)
        remote_path = os.path.relpath(os.path.join(root, nombre_file), proyecto_root).replace("\\", "/")

        resultado = actualizar_github_si_cambia(local_path, remote_path, f"Actualizar {remote_path}")

        if resultado in resumen:
            resumen[resultado].append(remote_path)
        else:
            resumen["errores"].append(remote_path)

# ------------------- Resumen final -------------------
print("\n📄 Resumen de subida al repo:")
for key, archivos in resumen.items():
    print(f"{key.upper()}: {len(archivos)} archivos")
    for a in archivos:
        print(f"   - {a}")
