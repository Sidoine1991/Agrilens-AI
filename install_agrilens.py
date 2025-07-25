import os
import sys
import subprocess
import tarfile
import requests

VENV_DIR = "venv"
REQUIREMENTS = "requirements.txt"

FR = sys.platform.startswith("win")

# 1. Créer l'environnement virtuel si besoin
def create_venv():
    if not os.path.isdir(VENV_DIR):
        print("[INFO] Création de l'environnement virtuel...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("[INFO] Environnement virtuel déjà présent.")

# 2. Installer les dépendances
def install_requirements():
    pip_path = os.path.join(VENV_DIR, "Scripts" if FR else "bin", "pip")
    print("[INFO] Installation des dépendances...")
    subprocess.check_call([pip_path, "install", "-r", REQUIREMENTS])

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def extract_tar(tar_path, extract_path):
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_path)

MODEL_DIR = "models/gemma-3n-transformers-gemma-3n-e2b-it-v1"
MODEL_TAR = "models/gemma-3n-transformers-gemma-3n-e2b-it-v1.tar"
GDRIVE_ID = "17WZeUKSxBHqFtfqm04MkAd7Ak6Yis-FM"

if not os.path.isdir(MODEL_DIR):
    os.makedirs("models", exist_ok=True)
    if not os.path.isfile(MODEL_TAR):
        print("Téléchargement du modèle depuis Google Drive...")
        download_file_from_google_drive(GDRIVE_ID, MODEL_TAR)
    print("Décompression du modèle...")
    extract_tar(MODEL_TAR, "models/")
    print("Modèle prêt dans:", MODEL_DIR)
else:
    print("Modèle déjà présent dans:", MODEL_DIR)

# 3. Vérifier la présence du modèle
def check_model():
    if not os.path.isdir(MODEL_DIR):
        print(f"[ERREUR] Le modèle Gemma 3n n'est pas trouvé dans {MODEL_DIR} !")
        print("[EN] Gemma 3n model not found in", MODEL_DIR)
        print("Veuillez placer le dossier du modèle téléchargé dans ce chemin avant de lancer l'application.")
        sys.exit(1)
    else:
        print("[OK] Modèle Gemma 3n trouvé.")

# 4. Instructions de lancement
def print_instructions():
    print("\n---")
    print("🇫🇷 Installation terminée ! Lancez l'application avec :")
    print(f"  {VENV_DIR}\\Scripts\\activate && streamlit run src/streamlit_app.py" if FR else f"  source {VENV_DIR}/bin/activate && streamlit run src/streamlit_app.py")
    print("\n🇬🇧 Installation complete! Launch the app with:")
    print(f"  {VENV_DIR}\\Scripts\\activate && streamlit run src/streamlit_app.py" if FR else f"  source {VENV_DIR}/bin/activate && streamlit run src/streamlit_app.py")
    print("---\n")

if __name__ == "__main__":
    create_venv()
    install_requirements()
    check_model()
    print_instructions() 