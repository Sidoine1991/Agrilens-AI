import os
import zipfile

# Nom de la compétition et chemin de destination
COMPETITION = "<competition-name>"  # Remplace par le nom exact de la compétition
DEST_DIR = "D:/Dev/AgriLensAI/models/"

# Crée le dossier s'il n'existe pas
os.makedirs(DEST_DIR, exist_ok=True)

# Commande pour télécharger le dataset de la compétition
os.system(f"kaggle competitions download -c {COMPETITION} -p {DEST_DIR}")

# Trouver et extraire tous les fichiers zip téléchargés
for file in os.listdir(DEST_DIR):
    if file.endswith(".zip"):
        zip_path = os.path.join(DEST_DIR, file)
        print(f"Extraction de {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DEST_DIR)
        print(f"Suppression de {zip_path}...")
        os.remove(zip_path)

print("Téléchargement et extraction terminés. Les modèles sont dans:", DEST_DIR)
