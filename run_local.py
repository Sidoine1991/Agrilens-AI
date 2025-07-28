#!/usr/bin/env python3
"""
Script de lancement pour AgriLens AI - Version Locale
Lance l'application avec toutes les fonctionnalités de la version Hugging Face
"""

import os
import sys
import subprocess

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
    """Lance l'application Streamlit"""
    
    print("🚀 Lancement d'AgriLens AI - Version Locale")
    print("📱 Interface web : http://localhost:8501")
    print("🔄 Appuyez sur Ctrl+C pour arrêter")
    print("-" * 50)
    
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
    
    print("🌱 AgriLens AI - Version Locale")
    print("=" * 40)
    
    # Configuration de l'environnement
    setup_environment()
    
    # Vérification des dépendances
    if not check_dependencies():
        return
    
    # Lancement de l'application
    launch_app()

if __name__ == "__main__":
    main() 