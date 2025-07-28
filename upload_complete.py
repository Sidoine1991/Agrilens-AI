#!/usr/bin/env python3
"""
Script complet pour uploader automatiquement tous les fichiers du modèle vers Hugging Face Datasets (adapté aux noms de fichiers réels)
"""

import os
import sys
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_model_files():
    """Upload tous les fichiers du modèle vers le dataset Hugging Face"""
    
    # Configuration
    repo_id = "Sidoineko/data_gemma"
    model_folder = "model_gemma"
    
    # Initialiser l'API
    api = HfApi()
    
    try:
        # Créer le dataset s'il n'existe pas
        print(f"🔧 Création/vérification du dataset {repo_id}...")
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=False
        )
        
        # Chemin vers les fichiers du modèle
        model_path = Path(".")
        
        # Liste des fichiers à uploader (adaptée aux noms réels)
        files_to_upload = [
            ("chat_template.jinja", "chat_template.jinja"),
            ("config.json", "config"),
            ("generation_config.json", "generation_config"),
            ("model.safetensors.index.json", "model.safetensors.index"),
            ("model-00001-of-00003.safetensors", "model-00001-of-00003.safetensors"),
            ("model-00002-of-00003.safetensors", "model-00002-of-00003.safetensors"),
            ("model-00003-of-00003.safetensors", "model-00003-of-00003.safetensors"),
            ("preprocessor_config.json", "preprocessor_config"),
            ("processor_config.json", "processor_config"),
            ("README.md", "README.md"),
            ("special_tokens_map.json", "special_tokens_map"),
            ("tokenizer.json", "tokenizer"),
            ("tokenizer.model", "tokenizer.model"),
            ("tokenizer_config.json", "tokenizer_config"),
        ]
        
        print(f"📁 Upload des fichiers vers {repo_id}/{model_folder}/")
        print("⏳ Cela peut prendre plusieurs minutes...")
        
        # Upload chaque fichier
        for i, (local_name, repo_name) in enumerate(files_to_upload, 1):
            file_path = model_path / local_name
            
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024 * 1024)  # Taille en MB
                print(f"⬆️  [{i}/{len(files_to_upload)}] Upload de {local_name} -> {repo_name} ({file_size:.1f} MB)...")
                
                # Upload avec le chemin complet dans le dataset
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"{model_folder}/{repo_name}",
                    repo_id=repo_id,
                    repo_type="dataset"
                )
                print(f"✅ {local_name} uploadé sous {repo_name}")
            else:
                print(f"⚠️  [{i}/{len(files_to_upload)}] Fichier {local_name} non trouvé, ignoré")
        
        print(f"\n🎉 Upload terminé !")
        print(f"📂 Vérifiez sur : https://huggingface.co/datasets/{repo_id}/tree/main/{model_folder}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'upload : {e}")
        print("💡 Assurez-vous d'être connecté avec : huggingface-cli login")
        return False
    
    return True

if __name__ == "__main__":
    print("🤖 Script d'upload automatique du modèle AgriLens (adapté)")
    print("=" * 50)
    
    # Vérifier que nous sommes dans le bon répertoire
    if not os.path.exists("config.json"):
        print("❌ Erreur : Fichier 'config.json' non trouvé")
        print("💡 Assurez-vous d'être dans le répertoire contenant les fichiers du modèle")
        sys.exit(1)
    
    # Vérifier la connexion Hugging Face
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✅ Connecté en tant que : {user}")
    except Exception as e:
        print("❌ Erreur de connexion à Hugging Face")
        print("💡 Lancez : huggingface-cli login")
        sys.exit(1)
    
    success = upload_model_files()
    
    if success:
        print("\n🎯 Prochaines étapes :")
        print("1. Vérifiez l'upload sur Hugging Face")
        print("2. Testez le modèle sur votre app Streamlit")
        print("3. Si tout fonctionne, vous pouvez supprimer les fichiers locaux")
    else:
        print("\n❌ L'upload a échoué. Vérifiez les erreurs ci-dessus.") 