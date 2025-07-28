#!/usr/bin/env python3
"""
Test rapide de l'application AgriLens AI - Version Locale
"""

import os
import sys

def test_imports():
    """Teste les imports nécessaires"""
    print("🔍 Test des imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit importé")
    except ImportError as e:
        print(f"❌ Erreur import Streamlit: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI importé")
    except ImportError as e:
        print(f"❌ Erreur import Google Generative AI: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ Transformers importé")
    except ImportError as e:
        print(f"❌ Erreur import Transformers: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch importé")
    except ImportError as e:
        print(f"❌ Erreur import PyTorch: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow importé")
    except ImportError as e:
        print(f"❌ Erreur import Pillow: {e}")
        return False
    
    return True

def test_configuration():
    """Teste la configuration"""
    print("\n🔧 Test de la configuration...")
    
    # Configuration des variables d'environnement
    os.environ['GOOGLE_API_KEY'] = "AIzaSyC4a4z20p7EKq1Fk5_AX8eB_1yBo1HqYvA"
    os.environ['HF_TOKEN'] = "hf_gUGRsgWffLNZVuzYLsmTdPwESIyrbryZW"
    
    google_key = os.getenv('GOOGLE_API_KEY')
    hf_token = os.getenv('HF_TOKEN')
    
    if google_key:
        print("✅ GOOGLE_API_KEY configurée")
    else:
        print("❌ GOOGLE_API_KEY manquante")
        return False
    
    if hf_token:
        print("✅ HF_TOKEN configuré")
    else:
        print("❌ HF_TOKEN manquant")
        return False
    
    return True

def test_gemini_config():
    """Teste la configuration Gemini"""
    print("\n🤖 Test de la configuration Gemini...")
    
    try:
        import google.generativeai as genai
        
        # Configuration
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Test de création du modèle
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Modèle Gemini créé avec succès")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur configuration Gemini: {e}")
        return False

def test_file_syntax():
    """Teste la syntaxe du fichier principal"""
    print("\n📝 Test de la syntaxe du fichier principal...")
    
    try:
        with open('src/streamlit_app_multilingual.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Test de compilation
        compile(content, 'src/streamlit_app_multilingual.py', 'exec')
        print("✅ Syntaxe du fichier principal correcte")
        return True
        
    except SyntaxError as e:
        print(f"❌ Erreur de syntaxe: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test de syntaxe: {e}")
        return False

def main():
    """Fonction principale"""
    print("🌱 AgriLens AI - Test Rapide de Configuration")
    print("=" * 50)
    
    # Tests
    imports_ok = test_imports()
    config_ok = test_configuration()
    gemini_ok = test_gemini_config()
    syntax_ok = test_file_syntax()
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    if imports_ok:
        print("✅ Imports : OK")
    else:
        print("❌ Imports : Problèmes détectés")
    
    if config_ok:
        print("✅ Configuration : OK")
    else:
        print("❌ Configuration : Problèmes détectés")
    
    if gemini_ok:
        print("✅ Gemini : OK")
    else:
        print("❌ Gemini : Problèmes détectés")
    
    if syntax_ok:
        print("✅ Syntaxe : OK")
    else:
        print("❌ Syntaxe : Problèmes détectés")
    
    # Recommandations
    print("\n💡 RECOMMANDATIONS")
    print("-" * 30)
    
    if all([imports_ok, config_ok, gemini_ok, syntax_ok]):
        print("🎉 Tous les tests sont passés !")
        print("🚀 L'application est prête à être lancée :")
        print("   streamlit run src/streamlit_app_multilingual.py --server.port=8501")
        print("   ou")
        print("   python run_local.py")
        print("   ou")
        print("   lancer_agrilens_local.bat")
    else:
        if not imports_ok:
            print("📦 Installez les dépendances manquantes :")
            print("   pip install -r requirements.txt")
        
        if not config_ok:
            print("🔑 Vérifiez la configuration des variables d'environnement")
        
        if not gemini_ok:
            print("🤖 Vérifiez la clé API Google Gemini")
        
        if not syntax_ok:
            print("📝 Corrigez les erreurs de syntaxe dans le fichier principal")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 