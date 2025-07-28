#!/usr/bin/env python3
"""
Script de test pour vérifier la configuration locale d'AgriLens AI
"""

import os
import sys

def test_environment():
    """Teste la configuration de l'environnement"""
    print("🔍 Test de l'environnement...")
    
    # Variables d'environnement
    google_key = os.getenv('GOOGLE_API_KEY')
    hf_token = os.getenv('HF_TOKEN')
    
    if google_key:
        print("✅ GOOGLE_API_KEY configurée")
    else:
        print("❌ GOOGLE_API_KEY manquante")
    
    if hf_token:
        print("✅ HF_TOKEN configuré")
    else:
        print("❌ HF_TOKEN manquant")
    
    return bool(google_key and hf_token)

def test_dependencies():
    """Teste les dépendances Python"""
    print("\n📦 Test des dépendances...")
    
    dependencies = [
        ('streamlit', 'Streamlit'),
        ('transformers', 'Transformers'),
        ('torch', 'PyTorch'),
        ('PIL', 'Pillow'),
        ('google.generativeai', 'Google Generative AI'),
        ('dotenv', 'Python-dotenv')
    ]
    
    all_ok = True
    
    for module, name in dependencies:
        try:
            __import__(module.replace('.', '_'))
            print(f"✅ {name} installé")
        except ImportError:
            print(f"❌ {name} manquant")
            all_ok = False
    
    return all_ok

def test_files():
    """Teste la présence des fichiers nécessaires"""
    print("\n📁 Test des fichiers...")
    
    required_files = [
        'src/streamlit_app_multilingual.py',
        'requirements.txt',
        'config_local.py',
        'run_local.py',
        'lancer_agrilens_local.bat'
    ]
    
    all_ok = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} présent")
        else:
            print(f"❌ {file_path} manquant")
            all_ok = False
    
    return all_ok

def setup_environment():
    """Configure l'environnement pour les tests"""
    print("\n🔧 Configuration de l'environnement...")
    
    # Définir les variables d'environnement
    os.environ['GOOGLE_API_KEY'] = "AIzaSyC4a4z20p7EKq1Fk5_AX8eB_1yBo1HqYvA"
    os.environ['HF_TOKEN'] = "hf_gUGRsgWffLNZVuzYLsmTdPwESIyrbryZW"
    
    print("✅ Variables d'environnement configurées")

def main():
    """Fonction principale"""
    print("🌱 AgriLens AI - Test de Configuration Locale")
    print("=" * 50)
    
    # Configuration
    setup_environment()
    
    # Tests
    env_ok = test_environment()
    deps_ok = test_dependencies()
    files_ok = test_files()
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    if env_ok:
        print("✅ Environnement : OK")
    else:
        print("❌ Environnement : Problèmes détectés")
    
    if deps_ok:
        print("✅ Dépendances : OK")
    else:
        print("❌ Dépendances : Problèmes détectés")
    
    if files_ok:
        print("✅ Fichiers : OK")
    else:
        print("❌ Fichiers : Problèmes détectés")
    
    # Recommandations
    print("\n💡 RECOMMANDATIONS")
    print("-" * 30)
    
    if not deps_ok:
        print("📦 Installez les dépendances :")
        print("   pip install -r requirements.txt")
    
    if not env_ok:
        print("🔑 Configurez les variables d'environnement")
    
    if not files_ok:
        print("📁 Vérifiez que tous les fichiers sont présents")
    
    if env_ok and deps_ok and files_ok:
        print("🎉 Configuration complète !")
        print("🚀 Lancez l'application avec :")
        print("   python run_local.py")
        print("   ou")
        print("   lancer_agrilens_local.bat")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 