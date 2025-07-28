#!/usr/bin/env python3
"""
Script pour uploader automatiquement tous les fichiers du modèle vers Hugging Face Datasets
"""

import os
import sys
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_model_files():
    """Upload tous les fichiers du modèle vers le dataset Hugging Face"""
    
    # Configuration
    dataset_name = "Sidoineko/data_gemma"
    model_folder = "model_gemma"
    
    # Initialiser l'API
    api = HfApi()
    
    try:
        # Créer le dataset s'il n'existe pas
        print(f"🔧 Création du dataset {dataset_name}...")
        create_repo(
            repo_id=dataset_name,
            repo_type="dataset",
            exist_ok=True,
            private=False
        )
        
        # Chemin vers les fichiers du modèle
        model_path = Path(".")
        
        # Liste des fichiers à uploader
        files_to_upload = [
            "chat_template.jinja",
            "config",
            "generation_config", 
            "model.safetensors.index",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors", 
            "model-00003-of-00003.safetensors",
            "preprocessor_config",
            "processor_config",
            "README.md",
            "special_tokens_map",
            "tokenizer",
            "tokenizer.model",
            "tokenizer_config"
        ]
        
        print(f"📁 Upload des fichiers vers {dataset_name}/{model_folder}/")
        
        # Upload chaque fichier
        for filename in files_to_upload:
            file_path = model_path / filename
            
            if file_path.exists():
                print(f"⬆️  Upload de {filename}...")
                
                # Upload avec le chemin complet dans le dataset
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"{model_folder}/{filename}",
                    repo_id=dataset_name,
                    repo_type="dataset"
                )
                print(f"✅ {filename} uploadé avec succès")
            else:
                print(f"⚠️  Fichier {filename} non trouvé, ignoré")
        
        print(f"\n🎉 Upload terminé !")
        print(f"📂 Vérifiez sur : https://huggingface.co/datasets/{dataset_name}/tree/main/{model_folder}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'upload : {e}")
        print("💡 Assurez-vous d'être connecté avec : huggingface-cli login")
        return False
    
    return True

if __name__ == "__main__":
    print("🤖 Script d'upload automatique du modèle AgriLens")
    print("=" * 50)
    
    # Vérifier que nous sommes dans le bon répertoire
    if not os.path.exists("config"):
        print("❌ Erreur : Fichier 'config' non trouvé")
        print("💡 Assurez-vous d'être dans le répertoire contenant les fichiers du modèle")
        sys.exit(1)
    
    success = upload_model_files()
    
    if success:
        print("\n🎯 Prochaines étapes :")
        print("1. Vérifiez l'upload sur Hugging Face")
        print("2. Testez le modèle sur votre app Streamlit")
        print("3. Si tout fonctionne, vous pouvez supprimer les fichiers locaux")
    else:
        print("\n❌ L'upload a échoué. Vérifiez les erreurs ci-dessus.") 