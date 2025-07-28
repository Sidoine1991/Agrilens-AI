#!/usr/bin/env python3
"""
Script de lancement pour AgriLens AI - Version Locale avec Modèle Local
Lance l'application avec le modèle Gemma 3n E4B IT depuis D:/Dev/model_gemma
"""

import os
import sys
import subprocess
from pathlib import Path

def check_local_model():
    """Vérifie que le modèle local existe"""
    model_path = Path("D:/Dev/model_gemma")
    
    if not model_path.exists():
        print("❌ Modèle local non trouvé dans D:/Dev/model_gemma")
        print("💡 Assurez-vous que le modèle Gemma 3n E4B IT est téléchargé dans ce dossier")
        return False
    
    # Vérifier la présence de fichiers essentiels
    required_files = ["config.json", "tokenizer.json", "model-00001-of-00003.safetensors"]
    missing_files = []
    
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Fichiers manquants dans le modèle local : {', '.join(missing_files)}")
        return False
    
    print("✅ Modèle local trouvé dans D:/Dev/model_gemma")
    return True

def setup_environment():
    """Configure l'environnement pour la version locale"""
    
    # Définir les variables d'environnement
    os.environ['GOOGLE_API_KEY'] = "AIzaSyC4a4z20p7EKq1Fk5_AX8eB_1yBo1HqYvA"
    os.environ['HF_TOKEN'] = "hf_gUGRsgWffLNZVuzYLsmTdPwESIyrbryZW"
    
    print("✅ Variables d'environnement configurées")
    print("🔑 Google API Key configurée")
    print("🤗 Hugging Face Token configuré")

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    
    required_packages = [
        'streamlit',
        'transformers',
        'torch',
        'PIL',
        'google-generativeai',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Packages manquants : {', '.join(missing_packages)}")
        print("💡 Installez-les avec : pip install -r requirements.txt")
        return False
    
    print("✅ Toutes les dépendances sont installées")
    return True

def launch_app():
    """Lance l'application Streamlit avec le modèle local"""
    
    print("🚀 Lancement d'AgriLens AI - Version Locale avec Modèle Local")
    print("📁 Modèle : D:/Dev/model_gemma")
    print("📱 Interface web : http://localhost:8501")
    print("🔄 Appuyez sur Ctrl+C pour arrêter")
    print("-" * 60)
    
    try:
        # Lancer l'application avec le fichier principal
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/streamlit_app_multilingual.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application arrêtée")
    except Exception as e:
        print(f"❌ Erreur lors du lancement : {e}")

def main():
    """Fonction principale"""
    
    print("🌱 AgriLens AI - Version Locale avec Modèle Local")
    print("=" * 50)
    
    # Vérification du modèle local
    if not check_local_model():
        return
    
    # Configuration de l'environnement
    setup_environment()
    
    # Vérification des dépendances
    if not check_dependencies():
        return
    
    # Lancement de l'application
    launch_app()

if __name__ == "__main__":
    main() 