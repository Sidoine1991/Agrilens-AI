from huggingface_hub import upload_folder, create_repo
import os

print("Début de l'upload vers Hugging Face Datasets...")

# Créer le dataset s'il n'existe pas
try:
    create_repo(
        repo_id="Sidoineko/data_gemma",
        repo_type="dataset",
        exist_ok=True
    )
    print("Dataset créé/vérifié avec succès")
except Exception as e:
    print(f"Erreur lors de la création du dataset: {e}")

# Upload tous les fichiers du dossier courant vers le dataset
upload_folder(
    folder_path=".",
    path_in_repo="model_gemma/",
    repo_type="dataset",
    repo_id="Sidoineko/data_gemma"
)

print("Upload terminé ! Vérifiez sur : https://huggingface.co/datasets/Sidoineko/data_gemma/tree/main/model_gemma") 