import os
import tarfile
import requests

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
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

MODEL_DIR = "models/gemma-3n-transformers-gemma-3n-e2b-it-v1"
MODEL_TAR = "models/gemma-3n-transformers-gemma-3n-e2b-it-v1.tar.gz"
GDRIVE_ID = "17WZeUKSxBHqFtfqm04MkAd7Ak6Yis-FM"  # ID du fichier .tar.gz

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