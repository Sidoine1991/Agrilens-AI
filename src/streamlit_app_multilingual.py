"""
================================================================================
AGRILENS AI - APPLICATION DE DIAGNOSTIC DES MALADIES DE PLANTES
================================================================================

Auteur: Sidoine Kolaolé YEBADOKPO
Email: syebadokpo@gmail.com
Localisation: Bohicon, République du Bénin
Version: 2.0 (avec mode mobile)
Date: Juillet 2025

DESCRIPTION:
-----------
AgriLens AI est une application d'intelligence artificielle conçue pour diagnostiquer
les maladies des plantes à partir d'images ou de descriptions textuelles. L'application
utilise le modèle Gemma 3n de Google pour analyser les symptômes et fournir des
recommandations de traitement.

FONCTIONNALITÉS PRINCIPALES:
---------------------------
1. Analyse d'images de plantes malades
2. Analyse de descriptions textuelles de symptômes
3. Interface multilingue (Français/English)
4. Mode mobile et desktop
5. Fonctionnement offline
6. Persistance du modèle en cache
7. Support pour Hugging Face Spaces

ARCHITECTURE TECHNIQUE:
----------------------
- Framework: Streamlit (Python)
- Modèle IA: Gemma 3n E4B IT (Google)
- Bibliothèques: Transformers, PyTorch, PIL
- Déploiement: Hugging Face Spaces + Local
- Interface: Responsive (Mobile/Desktop)

STRUCTURE DU CODE:
-----------------
1. Configuration et imports
2. Système de traduction multilingue
3. Gestion du mode mobile/desktop
4. Chargement et persistance des modèles
5. Analyse d'images et de texte
6. Interface utilisateur (onglets)
7. Documentation et manuel

UTILISATION:
-----------
1. Lancer: streamlit run src/streamlit_app_multilingual.py
2. Charger le modèle via la sidebar
3. Choisir le mode d'analyse (image ou texte)
4. Soumettre les données pour diagnostic
5. Consulter les résultats et recommandations

REQUIS SYSTÈME:
--------------
- Python 3.8+
- RAM: 8GB minimum (16GB recommandé)
- GPU: Optionnel (CUDA supporté)
- Espace disque: 5GB pour les modèles

LICENCE:
--------
Projet développé pour la Google - Gemma 3n Hackathon
Licence Creative Commons Attribution 4.0 International (CC BY 4.0)
Attribution : Sidoine Kolaolé YEBADOKPO
Usage éducatif, commercial et de démonstration autorisé

================================================================================
"""

import streamlit as st
import os
import io
from PIL import Image
import torch
import gc
import time
import sys
import psutil
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from huggingface_hub import HfFolder
from datetime import datetime

# Import optionnel pour Gemini (pour éviter les erreurs si pas installé)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- Configuration de la Page ---
# Configuration de base de l'interface Streamlit

# CSS personnalisé pour s'assurer que le menu hamburger soit visible
st.markdown("""
<style>
/* S'assurer que le menu hamburger soit visible */
#MainMenu {visibility: visible !important;}
footer {visibility: hidden;}
header {visibility: visible;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)
st.set_page_config(
    page_title="AgriLens AI - Diagnostic des Plantes",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# AgriLens AI\nApplication de diagnostic des maladies de plantes par IA"
    }
)

# Configuration pour forcer l'affichage du menu
st.markdown("""
<script>
// Forcer l'affichage du menu hamburger
document.addEventListener('DOMContentLoaded', function() {
    const menuButton = document.querySelector('[data-testid="baseButton-secondary"]');
    if (menuButton) {
        menuButton.style.display = 'block';
    }
});
</script>
""", unsafe_allow_html=True)

# --- Mode Mobile Detection et Configuration ---
def is_mobile():
    """
    Détecte si l'utilisateur est en mode mobile.
    
    Returns:
        bool: True si le mode mobile est activé, False sinon
    """
    return st.session_state.get('mobile_mode', False)

def toggle_mobile_mode():
    """
    Bascule entre le mode desktop et mobile.
    Met à jour l'état de session et déclenche un rechargement de l'interface.
    """
    if 'mobile_mode' not in st.session_state:
        st.session_state.mobile_mode = False
    st.session_state.mobile_mode = not st.session_state.mobile_mode

# --- CSS pour Mode Mobile ---
# Styles CSS personnalisés pour l'interface mobile et desktop
MOBILE_CSS = """
<style>
    /* Mode Mobile - Interface simulant un smartphone */
    .mobile-container {
        max-width: 375px !important;
        margin: 0 auto !important;
        border: 2px solid #ddd !important;
        border-radius: 20px !important;
        padding: 20px !important;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    /* Indicateur de notch pour smartphone */
    .mobile-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 4px;
        background: #333;
        border-radius: 0 0 10px 10px;
    }
    
    /* Header mobile avec gradient */
    .mobile-header {
        text-align: center;
        margin-bottom: 20px;
        padding: 10px;
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-radius: 15px;
        color: white;
        font-weight: bold;
    }
    
    /* Badge de statut offline */
    .mobile-status {
        display: inline-block;
        padding: 5px 10px;
        background: #28a745;
        color: white;
        border-radius: 20px;
        font-size: 12px;
        margin-top: 10px;
    }
    
    /* Boutons stylisés pour mobile */
    .mobile-button {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(0,123,255,0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .mobile-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,123,255,0.4) !important;
    }
    
    /* Champs de saisie arrondis */
    .mobile-input {
        border-radius: 15px !important;
        border: 2px solid #e9ecef !important;
        padding: 12px !important;
        background: white !important;
    }
    
    /* Conteneurs d'onglets */
    .mobile-tab {
        background: #f8f9fa !important;
        border-radius: 15px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    
    /* Styles pour les onglets en mode mobile */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 15px !important;
        background: #e9ecef !important;
        color: #495057 !important;
        font-weight: bold !important;
        padding: 8px 16px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        color: white !important;
    }
    
    /* Styles pour les boutons en mode mobile */
    .stButton > button {
        border-radius: 25px !important;
        font-weight: bold !important;
        padding: 12px 24px !important;
    }
    
    /* Styles pour les inputs en mode mobile */
    .stTextInput > div > div > input {
        border-radius: 15px !important;
        border: 2px solid #e9ecef !important;
        padding: 12px !important;
    }
    
    /* Styles pour les file uploaders en mode mobile */
    .stFileUploader > div {
        border-radius: 15px !important;
        border: 2px dashed #28a745 !important;
        background: #f8f9fa !important;
    }
    
    /* Mode Desktop - Interface classique */
    .desktop-container {
        max-width: 100% !important;
        margin: 0 auto !important;
        padding: 20px !important;
    }
    
    /* Design responsive pour petits écrans */
    @media (max-width: 768px) {
        .mobile-container {
            max-width: 100% !important;
            border-radius: 0 !important;
            margin: 0 !important;
        }
    }
</style>
"""

# Appliquer le CSS personnalisé
st.markdown(MOBILE_CSS, unsafe_allow_html=True)

# --- Système de Traduction Multilingue ---
# Dictionnaire contenant toutes les traductions de l'interface utilisateur
# Structure: {"clé": {"fr": "texte français", "en": "texte anglais"}}
TRANSLATIONS = {
    "title": {"fr": "AgriLens AI", "en": "AgriLens AI"},
    "subtitle": {"fr": "Votre assistant IA pour le diagnostic des maladies de plantes", "en": "Your AI Assistant for Plant Disease Diagnosis"},
    "config_title": {"fr": "Configuration", "en": "Configuration"},
    "load_model": {"fr": "Charger le modèle Gemma 3n E4B IT", "en": "Load Gemma 3n E4B IT Model"},
    "model_status": {"fr": "Statut du modèle :", "en": "Model Status:"},
    "not_loaded": {"fr": "Non chargé", "en": "Not loaded"},
    "loaded": {"fr": "✅ Chargé", "en": "✅ Loaded"},
    "error": {"fr": "❌ Erreur", "en": "❌ Error"},
    "tabs": {"fr": ["📸 Analyse d'Image", "💬 Analyse de Texte", "📖 Manuel", "ℹ️ À propos"], "en": ["📸 Image Analysis", "💬 Text Analysis", "📖 Manual", "ℹ️ About"]},
    "image_analysis_title": {"fr": "🔍 Diagnostic par Image", "en": "🔍 Image Diagnosis"},
    "image_analysis_desc": {"fr": "Téléchargez une photo de plante malade pour obtenir un diagnostic", "en": "Upload a photo of a diseased plant to get a diagnosis"},
    "choose_image": {"fr": "Choisissez une image...", "en": "Choose an image..."},
    "file_too_large_error": {"fr": "Erreur : Le fichier est trop volumineux. Maximum 200MB.", "en": "Error: File too large. Maximum 200MB."},
    "empty_file_error": {"fr": "Erreur : Le fichier est vide.", "en": "Error: File is empty."},
    "file_size_warning": {"fr": "Attention : Le fichier est très volumineux, le chargement peut prendre du temps.", "en": "Warning: File is very large, loading may take time."},
    "analyze_button": {"fr": "🔬 Analyser avec l'IA", "en": "🔬 Analyze with AI"},
    "analysis_results": {"fr": "## 📊 Résultats de l'Analyse", "en": "## 📊 Analysis Results"},
    "text_analysis_title": {"fr": "💬 Diagnostic par Texte", "en": "💬 Text Analysis"},
    "text_analysis_desc": {"fr": "Décrivez les symptômes de votre plante pour obtenir des conseils", "en": "Describe your plant's symptoms to get advice"},
    "symptoms_desc": {"fr": "Description des symptômes :", "en": "Symptom description:"},
    "manual_title": {"fr": "📖 Manuel d'utilisation", "en": "📖 User Manual"},
    "about_title": {"fr": "ℹ️ À propos d'AgriLens AI", "en": "ℹ️ About AgriLens AI"},
    "creator_title": {"fr": "👨‍💻 Créateur de l'Application", "en": "👨‍💻 Application Creator"},
    "creator_name": {"fr": "**Sidoine Kolaolé YEBADOKPO**", "en": "**Sidoine Kolaolé YEBADOKPO**"},
    "creator_location": {"fr": "Bohicon, République du Bénin", "en": "Bohicon, Benin Republic"},
    "creator_phone": {"fr": "+229 01 96 91 13 46", "en": "+229 01 96 91 13 46"},
    "creator_email": {"fr": "syebadokpo@gmail.com", "en": "syebadokpo@gmail.com"},
    "creator_linkedin": {"fr": "linkedin.com/in/sidoineko", "en": "linkedin.com/in/sidoineko"},
    "creator_portfolio": {"fr": "Hugging Face Portfolio: https://huggingface.co/spaces/Sidoineko/portfolio", "en": "Hugging Face Portfolio: https://huggingface.co/spaces/Sidoineko/portfolio"},
    "competition_title": {"fr": "🏆 Version Compétition Kaggle", "en": "🏆 Kaggle Competition Version"},
    "competition_text": {"fr": "Cette première version d'AgriLens AI a été développée spécifiquement pour participer à la compétition Kaggle. Elle représente notre première production publique et démontre notre expertise en IA appliquée à l'agriculture.", "en": "This first version of AgriLens AI was specifically developed to participate in the Kaggle competition. It represents our first public production and demonstrates our expertise in AI applied to agriculture."},
    "footer": {"fr": "*AgriLens AI - Diagnostic intelligent des plantes avec IA*", "en": "*AgriLens AI - Intelligent plant diagnosis with AI*"},
    "mobile_mode": {"fr": "📱 Mode Mobile", "en": "📱 Mobile Mode"},
    "desktop_mode": {"fr": "💻 Mode Desktop", "en": "💻 Desktop Mode"},
    "toggle_mode": {"fr": "🔄 Changer de mode", "en": "🔄 Toggle Mode"},
    "offline_status": {"fr": "Mode: OFFLINE", "en": "Mode: OFFLINE"},
    "online_status": {"fr": "Mode: ONLINE", "en": "Mode: ONLINE"},
    "mobile_demo": {"fr": "Démo App Mobile", "en": "Mobile App Demo"},
    "mobile_desc": {"fr": "Interface simulant l'application mobile offline", "en": "Interface simulating the offline mobile application"},
    "language_selection": {"fr": "Sélectionnez votre langue :", "en": "Select your language:"},
    "language_help": {"fr": "Change la langue de l'interface et des réponses de l'IA.", "en": "Changes the language of the interface and AI responses."},
    "hf_token_title": {"fr": "🔑 Jeton Hugging Face", "en": "🔑 Hugging Face Token"},
    "hf_token_found": {"fr": "✅ Jeton HF trouvé et configuré.", "en": "✅ HF token found and configured."},
    "hf_token_not_found": {"fr": "⚠️ Jeton HF non trouvé.", "en": "⚠️ HF token not found."},
    "hf_token_info": {"fr": "Il est recommandé de définir la variable d'environnement `HF_TOKEN` avec votre jeton personnel Hugging Face pour éviter les erreurs d'accès (403).", "en": "It is recommended to set the `HF_TOKEN` environment variable with your personal Hugging Face token to avoid access errors (403)."},
    "get_hf_token": {"fr": "Obtenir un jeton HF", "en": "Get HF token"},
    "model_title": {"fr": "🤖 Modèle IA Gemma 3n", "en": "🤖 Gemma 3n AI Model"},
    "model_loaded": {"fr": "✅ Modèle chargé", "en": "✅ Model loaded"},
    "model_not_loaded": {"fr": "❌ Modèle non chargé.", "en": "❌ Model not loaded."},
    "load_time": {"fr": "Heure de chargement : ", "en": "Load time: "},
    "device_used": {"fr": "Device utilisé : ", "en": "Device used: "},
    "reload_model": {"fr": "🔄 Recharger le modèle", "en": "🔄 Reload model"},
    "force_persistence": {"fr": "💾 Forcer Persistance", "en": "💾 Force Persistence"},
    "persistence_success": {"fr": "Persistance forcée avec succès.", "en": "Persistence forced successfully."},
    "persistence_failed": {"fr": "Échec de la persistance.", "en": "Persistence failed."},
    "loading_model": {"fr": "Chargement du modèle en cours...", "en": "Loading model..."},
    "model_loaded_success": {"fr": "✅ Modèle chargé avec succès !", "en": "✅ Model loaded successfully!"},
    "model_load_failed": {"fr": "❌ Échec du chargement du modèle.", "en": "❌ Model loading failed."},
    "persistence_title": {"fr": "💾 Persistance du Modèle", "en": "💾 Model Persistence"},
    "persistence_loaded": {"fr": "✅ Modèle chargé et persistant en cache.", "en": "✅ Model loaded and persistent in cache."},
    "persistence_warning": {"fr": "⚠️ Modèle chargé mais non persistant. Cliquez sur 'Forcer Persistance'.", "en": "⚠️ Model loaded but not persistent. Click 'Force Persistence'."},
    "persistence_not_loaded": {"fr": "⚠️ Modèle non chargé.", "en": "⚠️ Model not loaded."},
    "upload_image": {"fr": "📁 Upload d'image", "en": "📁 Upload image"},
    "webcam_capture": {"fr": "📷 Capture par webcam", "en": "📷 Webcam capture"},
    "choose_method": {"fr": "Choisissez votre méthode :", "en": "Choose your method:"},
    "webcam_title": {"fr": "**📷 Capture d'image par webcam**", "en": "**📷 Webcam image capture**"},
    "image_info_title": {"fr": "**Informations de l'image :**", "en": "**Image information:**"},
    "format_label": {"fr": "• Format : ", "en": "• Format: "},
    "original_size": {"fr": "• Taille originale : ", "en": "• Original size: "},
    "current_size": {"fr": "• Taille actuelle : ", "en": "• Current size: "},
    "mode_label": {"fr": "• Mode : ", "en": "• Mode: "},
    "pixels": {"fr": " pixels", "en": " pixels"},
    "symptoms_label": {"fr": "Description des symptômes :", "en": "Symptom description:"},
    "mission_title": {"fr": "### 🌱 Notre Mission / Our Mission", "en": "### 🌱 Our Mission"},
    "mission_text": {"fr": "AgriLens AI est une application de diagnostic des maladies de plantes utilisant l'intelligence artificielle pour aider les agriculteurs à identifier et traiter les problèmes de leurs cultures.", "en": "AgriLens AI is a plant disease diagnosis application using artificial intelligence to help farmers identify and treat problems with their crops."},
    "features_title": {"fr": "### 🚀 Fonctionnalités / Features", "en": "### 🚀 Features"},
    "features_text": {"fr": "• **Analyse d'images** : Diagnostic visuel des maladies\n• **Analyse de texte** : Conseils basés sur les descriptions\n• **Recommandations pratiques** : Actions concrètes à entreprendre\n• **Interface optimisée** : Pour une utilisation sur divers appareils\n• **Support multilingue** : Français et Anglais", "en": "• **Image analysis** : Visual diagnosis of diseases\n• **Text analysis** : Advice based on descriptions\n• **Practical recommendations** : Concrete actions to take\n• **Optimized interface** : For use on various devices\n• **Multilingual support** : French and English"},
    "technology_title": {"fr": "### 🔧 Technologie / Technology", "en": "### 🔧 Technology"},
    "local_model_text": {"fr": "• **Modèle** : Gemma 3n E4B IT (Local - {path})\n• **Framework** : Streamlit\n• **Déploiement** : Local", "en": "• **Model** : Gemma 3n E4B IT (Local - {path})\n• **Framework** : Streamlit\n• **Deployment** : Local"},
    "online_model_text": {"fr": "• **Modèle** : Gemma 3n E4B IT (Hugging Face - en ligne)\n• **Framework** : Streamlit\n• **Déploiement** : Hugging Face Spaces", "en": "• **Model** : Gemma 3n E4B IT (Hugging Face - online)\n• **Framework** : Streamlit\n• **Deployment** : Hugging Face Spaces"},
    "warning_title": {"fr": "### ⚠️ Avertissement / Warning", "en": "### ⚠️ Warning"},
    "warning_text": {"fr": "Les résultats fournis par l'IA sont à titre indicatif uniquement et ne remplacent pas l'avis d'un expert agricole qualifié.", "en": "The results provided by AI are for guidance only and do not replace the advice of a qualified agricultural expert."},
    "support_title": {"fr": "### 📞 Support", "en": "### 📞 Support"},
    "support_text": {"fr": "Pour toute question ou problème, consultez la documentation ou contactez le créateur.", "en": "For any questions or issues, consult the documentation or contact the creator."},
    "settings_button": {"fr": "⚙️ Réglages", "en": "⚙️ Settings"},
    "specific_question": {"fr": "Question spécifique (optionnel) :", "en": "Specific question (optional):"},
    "question_placeholder": {"fr": "Ex: Les feuilles ont des taches jaunes, que faire ?", "en": "Ex: The leaves have yellow spots, what to do?"},
    "webcam_info": {"fr": "💡 Positionnez votre plante malade devant la webcam et cliquez sur 'Prendre une photo'. Assurez-vous d'un bon éclairage.", "en": "💡 Position your sick plant in front of the webcam and click 'Take a photo'. Make sure you have good lighting."},
    "take_photo": {"fr": "Prendre une photo de la plante", "en": "Take a photo of the plant"},
    "image_processing_error": {"fr": "❌ Erreur lors du traitement de l'image uploadée : ", "en": "❌ Error processing uploaded image: "},
    "image_processing_error_webcam": {"fr": "❌ Erreur lors du traitement de l'image capturée : ", "en": "❌ Error processing captured image: "},
    "try_different_image": {"fr": "💡 Essayez avec une image différente ou un format différent (PNG, JPG, JPEG).", "en": "💡 Try with a different image or format (PNG, JPG, JPEG)."},
    "try_retake_photo": {"fr": "💡 Essayez de reprendre la photo.", "en": "💡 Try taking the photo again."},
    "image_resized_warning": {"fr": "⚠️ L'image a été redimensionnée de  à {new_size} pour optimiser le traitement.", "en": "⚠️ Image has been resized from  to {new_size} to optimize processing."},
    "model_not_loaded_error": {"fr": "❌ Modèle non chargé. Veuillez le charger dans les réglages.", "en": "❌ Model not loaded. Please load it in settings."},
    "analyzing_image": {"fr": "🔍 Analyse d'image en cours...", "en": "🔍 Analyzing image..."},
    "image_processing_general_error": {"fr": "Erreur lors du traitement de l'image : ", "en": "Error processing image: "},
    "symptoms_placeholder": {"fr": "Ex: Mes tomates ont des taches brunes sur les feuilles et les fruits, une poudre blanche sur les tiges...", "en": "Ex: My tomatoes have brown spots on leaves and fruits, white powder on stems..."},
    "culture_clarification": {"fr": "🌱 Clarification de la Culture", "en": "🌱 Culture Clarification"},
    "culture_question": {"fr": "Quelle est la culture concernée ?", "en": "What is the crop concerned?"},
    "culture_placeholder": {"fr": "Ex: Tomate, Piment, Maïs, Haricot, Aubergine...", "en": "Ex: Tomato, Pepper, Corn, Bean, Eggplant..."},
    "culture_help": {"fr": "Précisez le type de plante pour un diagnostic plus précis", "en": "Specify the plant type for more accurate diagnosis"},
    "diagnosis_with_culture": {"fr": "🔬 Diagnostic avec Culture Spécifiée", "en": "🔬 Diagnosis with Specified Culture"},
    "culture_specified": {"fr": "Culture spécifiée : ", "en": "Specified culture: "},
    "export_diagnostic": {"fr": "📄 Exporter le Diagnostic", "en": "📄 Export Diagnosis"},
    "export_html": {"fr": "💻 Exporter en HTML", "en": "💻 Export as HTML"},
    "export_text": {"fr": "📝 Exporter en Texte", "en": "📝 Export as Text"},
    "download_html": {"fr": "Télécharger HTML", "en": "Download HTML"},
    "download_text": {"fr": "Télécharger Texte", "en": "Download Text"},
    "export_success": {"fr": "✅ Diagnostic exporté avec succès !", "en": "✅ Diagnosis exported successfully!"},
    "export_error": {"fr": "❌ Erreur lors de l'export", "en": "❌ Export error"},
    "html_filename": {"fr": "diagnostic_agrilens_{date}.html", "en": "agrilens_diagnosis_{date}.html"},
    "text_filename": {"fr": "diagnostic_agrilens_{date}.txt", "en": "agrilens_diagnosis_{date}.txt"}
}

def t(key):
    """
    Fonction de traduction simple pour l'interface multilingue.
    
    Args:
        key (str): Clé de traduction à rechercher dans le dictionnaire TRANSLATIONS
        
    Returns:
        str: Texte traduit dans la langue actuelle, ou la clé si la traduction n'existe pas
        
    Example:
        >>> t("title")  # Retourne "AgriLens AI" en français ou anglais selon la langue active
    """
    if 'language' not in st.session_state:
        st.session_state.language = 'fr'
    lang = st.session_state.language
    return TRANSLATIONS.get(key, {}).get(lang, key)

# --- Initialisation de la langue ---
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

# --- Cache global pour le modèle ---
if 'global_model_cache' not in st.session_state:
    st.session_state.global_model_cache = {}
if 'model_load_time' not in st.session_state:
    st.session_state.model_load_time = None
if 'model_persistence_check' not in st.session_state:
    st.session_state.model_persistence_check = False

def check_model_persistence():
    """
    Vérifie si le modèle est toujours persistant en mémoire et fonctionnel.
    
    Cette fonction contrôle si le modèle chargé est toujours disponible
    et accessible dans la session Streamlit, ce qui est important pour
    éviter les rechargements inutiles.
    
    Returns:
        bool: True si le modèle est persistant et fonctionnel, False sinon
        
    Note:
        Cette vérification est particulièrement importante sur Hugging Face Spaces
        où la mémoire peut être limitée et les modèles peuvent être déchargés.
    """
    try:
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            if hasattr(st.session_state.model, 'device'):
                device = st.session_state.model.device
                return True
        return False
    except Exception:
        return False

def force_model_persistence():
    """
    Force la persistance du modèle et du processeur dans le cache global.
    
    Cette fonction sauvegarde explicitement le modèle et le processeur
    dans le cache de session pour garantir leur disponibilité entre
    les interactions utilisateur.
    
    Returns:
        bool: True si la persistance a réussi, False sinon
        
    Note:
        Cette fonction est utile pour éviter les rechargements coûteux
        du modèle, particulièrement important pour les modèles volumineux
        comme Gemma 3n.
    """
    try:
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            # Sauvegarder le modèle et le processeur
            st.session_state.global_model_cache['model'] = st.session_state.model
            st.session_state.global_model_cache['processor'] = st.session_state.processor
            st.session_state.global_model_cache['load_time'] = time.time()
            st.session_state.global_model_cache['model_type'] = type(st.session_state.model).__name__
            st.session_state.global_model_cache['processor_type'] = type(st.session_state.processor).__name__
            
            # Sauvegarder les informations sur le device
            if hasattr(st.session_state.model, 'device'):
                st.session_state.global_model_cache['device'] = st.session_state.model.device

            # Vérifier que la sauvegarde a réussi
            if st.session_state.global_model_cache.get('model') is not None:
                st.session_state.model_persistence_check = True
                return True
        return False
    except Exception:
        return False

def restore_model_from_cache():
    """Restaure le modèle et le processeur depuis le cache global."""
    try:
        if 'model' in st.session_state.global_model_cache and st.session_state.global_model_cache['model'] is not None:
            cached_model = st.session_state.global_model_cache['model']
            if hasattr(cached_model, 'device'):
                st.session_state.model = cached_model
                st.session_state.processor = st.session_state.global_model_cache.get('processor')
                st.session_state.model_loaded = True
                st.session_state.model_status = "Chargé (cache)"
                if 'load_time' in st.session_state.global_model_cache:
                    st.session_state.model_load_time = st.session_state.global_model_cache['load_time']
                return True
        return False
    except Exception:
        return False

def diagnose_loading_issues():
    """Diagnostique les problèmes potentiels de chargement (dépendances, ressources, etc.)."""
    issues = []
    
    if not HfFolder.get_token() and not os.environ.get("HF_TOKEN"):
        issues.append("⚠️ **Jeton Hugging Face (HF_TOKEN) non configuré.** Le téléchargement du modèle pourrait échouer ou être ralenti.")

    try:
        import transformers; issues.append(f"✅ Transformers v{transformers.__version__}")
        import torch; issues.append(f"✅ PyTorch v{torch.__version__}")
        if torch.cuda.is_available(): issues.append(f"✅ CUDA disponible : {torch.cuda.get_device_name(0)}")
        else: issues.append("⚠️ CUDA non disponible - utilisation CPU (plus lent)")
    except ImportError as e: issues.append(f"❌ Dépendance manquante : ")

    try:
        mem = psutil.virtual_memory()
        issues.append(f"💾 RAM disponible : {mem.available // (1024**3)} GB")
        if mem.available < 4 * 1024**3:
            issues.append("⚠️ RAM insuffisante (< 4GB) - Le chargement risque d'échouer.")
    except ImportError: issues.append("⚠️ Impossible de vérifier la mémoire système")

    return issues

def resize_image_if_needed(image, max_size=(800, 800)):
    """Redimensionne une image PIL si elle dépasse `max_size` tout en conservant les proportions."""
    width, height = image.size
    if width > max_size[0] or height > max_size[1]:
        ratio = min(max_size[0] / width, max_size[1] / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image, True
    return image, False

def afficher_ram_disponible(context=""):
    """Affiche l'utilisation de la RAM."""
    try:
        mem = psutil.virtual_memory()
        st.info(f"💾 RAM : {mem.available // (1024**3)} GB disponible")
        if mem.available < 4 * 1024**3:
            st.warning("⚠️ Moins de 4GB de RAM disponible, le chargement du modèle risque d'échouer !")
    except ImportError:
        st.warning("⚠️ Impossible de vérifier la RAM système.")

def generate_html_diagnostic(diagnostic_text, culture=None, image_info=None, timestamp=None):
    """
    Génère un fichier HTML formaté pour le diagnostic.
    
    Args:
        diagnostic_text (str): Le texte du diagnostic
        culture (str): La culture spécifiée
        image_info (dict): Informations sur l'image
        timestamp (str): Horodatage de l'analyse
        
    Returns:
        str: Contenu HTML formaté
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%d/%m/%Y à %H:%M")
    
    html_content = f"""
<!DOCTYPE html>
<html lang="{'fr' if st.session_state.language == 'fr' else 'en'}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgriLens AI - Diagnostic</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #4CAF50;
        }}
        .logo {{
            font-size: 2.5em;
            color: #4CAF50;
            margin-bottom: 10px;
        }}
        .title {{
            color: #333;
            font-size: 1.8em;
            margin-bottom: 5px;
        }}
        .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        .info-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 4px solid #4CAF50;
        }}
        .info-title {{
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 10px;
        }}
        .diagnostic-content {{
            background: #fff;
            padding: 25px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            margin-bottom: 20px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
            font-size: 0.9em;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
        .warning {{
            background: #f8d7da;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #dc3545;
            margin: 15px 0;
        }}
        .success {{
            background: #d4edda;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">🌱</div>
            <h1 class="title">AgriLens AI</h1>
            <p class="subtitle">Diagnostic Intelligent des Plantes</p>
        </div>
        
        <div class="info-section">
            <div class="info-title">📋 Informations de l'Analyse</div>
            <p><strong>Date et heure :</strong> {timestamp}</p>
            <p><strong>Modèle utilisé :</strong> Gemma 3n E4B IT</p>
            {f'<p><strong>Culture analysée :</strong> {culture}</p>' if culture else ''}
            {f'<p><strong>Format image :</strong> {image_info.get("format", "N/A")}</p>' if image_info else ''}
            {f'<p><strong>Taille image :</strong> {image_info.get("size", "N/A")}</p>' if image_info else ''}
        </div>
        
        <div class="diagnostic-content">
            <h2>🔬 Résultats du Diagnostic</h2>
            {diagnostic_text.replace('##', '<h3>').replace('**', '<strong>').replace('*', '<em>')}
        </div>
        
        <div class="highlight">
            <strong>⚠️ Avertissement :</strong> Ce diagnostic est fourni à titre indicatif. 
            Pour des cas critiques, consultez un expert agricole qualifié.
        </div>
        
        <div class="footer">
            <p>Généré par AgriLens AI - Créé par Sidoine Kolaolé YEBADOKPO</p>
            <p>Bohicon, République du Bénin | syebadokpo@gmail.com</p>
        </div>
    </div>
</body>
</html>
"""
    return html_content

def generate_text_diagnostic(diagnostic_text, culture=None, image_info=None, timestamp=None):
    """
    Génère un fichier texte formaté pour le diagnostic.
    
    Args:
        diagnostic_text (str): Le texte du diagnostic
        culture (str): La culture spécifiée
        image_info (dict): Informations sur l'image
        timestamp (str): Horodatage de l'analyse
        
    Returns:
        str: Contenu texte formaté
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%d/%m/%Y à %H:%M")
    
    text_content = f"""
================================================================================
AGRILENS AI - DIAGNOSTIC INTELLIGENT DES PLANTES
================================================================================

📋 INFORMATIONS DE L'ANALYSE
---------------------------
Date et heure : {timestamp}
Modèle utilisé : Gemma 3n E4B IT
{f'Culture analysée : {culture}' if culture else ''}
{f'Format image : {image_info.get("format", "N/A")}' if image_info else ''}
{f'Taille image : {image_info.get("size", "N/A")}' if image_info else ''}

🔬 RÉSULTATS DU DIAGNOSTIC
-------------------------
{diagnostic_text}

⚠️ AVERTISSEMENT
---------------
Ce diagnostic est fourni à titre indicatif. Pour des cas critiques, 
consultez un expert agricole qualifié.

================================================================================
Généré par AgriLens AI
Créé par Sidoine Kolaolé YEBADOKPO
Bohicon, République du Bénin
Email : syebadokpo@gmail.com
================================================================================
"""
    return text_content

# --- Fonctions d'Analyse avec Gemma 3n E4B IT ---
MODEL_ID_HF = "google/gemma-3n-E4B-it"
LOCAL_MODEL_PATH = "D:/Dev/model_gemma" # Chemin vers votre modèle local (ajustez si nécessaire)

def load_model_strategy(model_identifier, device_map=None, torch_dtype=None, quantization=None, force_persistence=False):
    """
    Charge un modèle et son processeur en utilisant des paramètres spécifiques.
    Retourne le modèle et le processeur, ou (None, None) en cas d'échec.
    """
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        common_args = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "token": os.environ.get("HF_TOKEN")
        }
        
        if quantization == "4bit":
            common_args.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16
            })
        elif quantization == "8bit":
            common_args.update({"load_in_8bit": True})
        
        processor = AutoProcessor.from_pretrained(model_identifier, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
        model = Gemma3nForConditionalGeneration.from_pretrained(model_identifier, **common_args)
        
        afficher_ram_disponible("après chargement")
        
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.model_loaded = True
        st.session_state.model_status = f"Chargé ({device_map or 'auto'})"
        st.session_state.model_load_time = time.time()
        
        if force_persistence:
            if force_model_persistence():
                st.success("Modèle chargé et persistant.")
            else:
                st.warning("Modèle chargé mais problème de persistance.")
        
        return model, processor

    except ImportError as e:
        raise ImportError(f"Bibliothèque manquante : . Installez-la avec `pip install transformers torch accelerate bitsandbytes`")
    except Exception as e:
        raise Exception(f"Échec du chargement avec la stratégie : {e}")

def load_model():
    """Charge le modèle avec une stratégie adaptative basée sur l'environnement."""
    try:
        st.info("🔍 Début du processus de chargement du modèle...")
        
        # Détecter l'environnement Hugging Face Spaces
        is_hf_spaces = os.environ.get('SPACE_ID') is not None
        st.info(f"🌍 Environnement détecté : {'Hugging Face Spaces' if is_hf_spaces else 'Local'}")
        
        if is_hf_spaces:
            st.info("🌐 Environnement Hugging Face Spaces détecté - Utilisation de la stratégie optimisée")
            result = load_ultra_lightweight_for_hf_spaces()
            st.info(f"📊 Résultat du chargement HF Spaces : {result[0] is not None and result[1] is not None}")
            return result
        else:
            st.info("💻 Environnement local détecté - Chargement du modèle Gemma 3n complet")
            result = load_gemma_full()
            st.info(f"📊 Résultat du chargement local : {result[0] is not None and result[1] is not None}")
            return result
            
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        st.error(f"🔍 Type d'erreur : {type(e).__name__}")
        return None, None

def load_ultra_lightweight_for_hf_spaces():
    """Charge un modèle léger pour Hugging Face Spaces (16GB RAM limit)"""
    try:
        st.info("🔄 Début du chargement du modèle Gemma 3B IT pour Hugging Face Spaces...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Charger Gemma 3B IT (plus léger que Gemma 3n E4B IT)
        model_id = "google/gemma-3b-it"
        st.info(f"📦 Modèle cible : {model_id}")
        
        st.info("🔧 Configuration ultra-légère en cours...")
        
        # Configuration ultra-légère
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        st.info("✅ Modèle chargé, chargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if model and tokenizer:
            st.success("✅ Modèle Gemma 3B IT chargé avec succès pour Hugging Face Spaces")
            st.info(f"📊 Modèle type : {type(model).__name__}")
            st.info(f"📊 Tokenizer type : {type(tokenizer).__name__}")
            return model, tokenizer
        else:
            st.error("❌ Échec du chargement du modèle léger")
            return None, None
            
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle léger : {str(e)}")
        st.error(f"🔍 Type d'erreur : {type(e).__name__}")
        return None, None

def load_gemma_full():
    """Charge le modèle Gemma 3n E4B IT complet pour les environnements locaux"""
    try:
        st.info("🔄 Chargement du modèle Gemma 3n E4B IT complet...")
        
        # Configuration pour environnement local avec plus de ressources
        model = Gemma3nForConditionalGeneration.from_pretrained(
            MODEL_ID_HF,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(MODEL_ID_HF)
        
        if model and processor:
            st.success("✅ Modèle Gemma 3n E4B IT chargé avec succès")
            return model, processor
        else:
            st.error("❌ Échec du chargement du modèle complet")
            return None, None
            
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle complet : {str(e)}")
        return None, None

def analyze_image_multilingual(image, prompt=""):
    """
    Analyse une image avec le modèle chargé (Gemma 3B IT ou Gemma 3n) pour un diagnostic précis.
    """
    if not st.session_state.model_loaded:
        if not restore_model_from_cache():
            st.warning("Modèle non chargé. Veuillez le charger via les réglages avant d'analyser.")
            return "❌ Modèle Gemma non chargé. Veuillez d'abord charger le modèle dans les réglages."
        else:
            st.info("Modèle restauré depuis le cache pour l'analyse.")

    model, processor = st.session_state.model, st.session_state.processor
    if not model or not processor:
        return "❌ Modèle Gemma non disponible. Veuillez recharger le modèle."

    try:
        # Détecter le type de modèle chargé
        is_gemma3b = "gemma-3b" in str(type(model)).lower()
        
        if is_gemma3b:
            # Utiliser la logique pour Gemma 3B IT (modèle léger)
            return analyze_with_gemma3b_and_gemini(image, prompt)
        else:
            # Utiliser la logique pour Gemma 3n (modèle complet)
            return analyze_with_gemma3n(image, prompt)
            
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
            return "❌ Erreur d'accès Hugging Face (403). Vérifiez votre HF_TOKEN."
        elif "out of memory" in error_msg.lower():
            return "❌ Erreur de mémoire insuffisante. Essayez de recharger le modèle."
        else:
            return f"❌ Erreur lors de l'analyse : {error_msg}"

def analyze_with_gemma3b_and_gemini(image, prompt=""):
    """Analyse avec Gemma 3B IT + Gemini pour Hugging Face Spaces"""
    try:
        model, tokenizer = st.session_state.model, st.session_state.processor
        
        # Préparer le prompt pour Gemma 3B IT
        if st.session_state.language == "fr":
            if "Culture spécifiée :" in prompt:
                user_prompt = f"Analyse cette image de plante en te concentrant sur la culture mentionnée. {prompt} Fournis un diagnostic précis."
            else:
                user_prompt = f"Analyse cette image de plante et fournis un diagnostic précis. Question : {prompt}" if prompt else "Analyse cette image de plante et fournis un diagnostic précis."
        else:
            if "Culture spécifiée :" in prompt:
                user_prompt = f"Analyze this plant image focusing on the mentioned crop. {prompt} Provide a precise diagnosis."
            else:
                user_prompt = f"Analyze this plant image and provide a precise diagnosis. Question: {prompt}" if prompt else "Analyze this plant image and provide a precise diagnosis."
        
        # Créer le prompt avec l'image
        final_prompt = f"<image>\n{user_prompt}"
        
        # Préparer les entrées
        inputs = tokenizer(text=final_prompt, images=image, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        
        # Générer la réponse
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            response = tokenizer.decode(generation[0][input_len:], skip_special_tokens=True)
        
        final_response = response.strip()
        
        # Améliorer avec Gemini si disponible
        if GEMINI_AVAILABLE and os.environ.get('GOOGLE_API_KEY'):
            try:
                genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
                model_gemini = genai.GenerativeModel('gemini-1.5-flash')
                
                gemini_prompt = f"""
                Tu es un expert en pathologie végétale. Améliore ce diagnostic de maladie de plante :
                
                {final_response}
                
                Fournis une version améliorée avec :
                1. Diagnostic précis
                2. Causes possibles
                3. Symptômes détaillés
                4. Traitements recommandés
                5. Niveau d'urgence
                
                Réponds en {st.session_state.language}.
                """
                
                gemini_response = model_gemini.generate_content(gemini_prompt)
                final_response = gemini_response.text
            except Exception as e:
                st.warning(f"Gemini non disponible : {str(e)}")
                pass  # Continuer sans Gemini si erreur
        
        if st.session_state.language == "fr":
            return f"""
## 🧠 **Analyse par Gemma 3B IT + Gemini**

{final_response}
"""
        else:
            return f"""
## 🧠 **Analysis by Gemma 3B IT + Gemini**

{final_response}
"""
            
    except Exception as e:
        return f"❌ Erreur lors de l'analyse avec Gemma 3B IT : {str(e)}"

def analyze_with_gemma3n(image, prompt=""):
    """Analyse avec Gemma 3n E4B IT (modèle complet)"""
    try:
        model, processor = st.session_state.model, st.session_state.processor
        
        # Préparer le prompt pour Gemma 3n
        if st.session_state.language == "fr":
            if "Culture spécifiée :" in prompt:
                user_instruction = f"Analyse cette image de plante en te concentrant spécifiquement sur la culture mentionnée. {prompt} Fournis un diagnostic précis et adapté à cette culture."
            else:
                user_instruction = f"Analyse cette image de plante et fournis un diagnostic précis. Question spécifique : {prompt}" if prompt else "Analyse cette image de plante et fournis un diagnostic précis."
            system_message = "Tu es un expert en pathologie végétale spécialisé dans le diagnostic des maladies de plantes. Réponds de manière structurée et précise, en incluant diagnostic, causes, symptômes, traitement et urgence. Si une culture spécifique est mentionnée, concentre-toi sur les maladies typiques de cette culture."
        else:
            if "Culture spécifiée :" in prompt:
                user_instruction = f"Analyze this plant image focusing specifically on the mentioned crop. {prompt} Provide a precise diagnosis adapted to this crop."
            else:
                user_instruction = f"Analyze this plant image and provide a precise diagnosis. Specific question: {prompt}" if prompt else "Analyze this plant image and provide a precise diagnosis."
            system_message = "You are an expert in plant pathology specialized in plant disease diagnosis. Respond in a structured and precise manner, including diagnosis, causes, symptoms, treatment, and urgency. If a specific crop is mentioned, focus on diseases typical of that crop."
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_instruction}
            ]}
        ]
        
        # Utiliser processor.apply_chat_template
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        device = getattr(model, 'device', 'cpu')
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                pixel_values=inputs["pixel_values"].to(device) if "pixel_values" in inputs else None,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            response = processor.decode(generation[0][input_len:], skip_special_tokens=True)

        final_response = response.strip()
        final_response = final_response.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()
        
        if st.session_state.language == "fr":
            return f"""
## 🧠 **Analyse par Gemma 3n E4B IT**

{final_response}
"""
        else:
            return f"""
## 🧠 **Analysis by Gemma 3n E4B IT**

{final_response}
"""
            
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
            return "❌ Erreur 403 - Accès refusé. Veuillez vérifier votre jeton Hugging Face (HF_TOKEN) et les quotas."
        elif "Number of images does not match number of special image tokens" in error_msg:
            return f"❌ Erreur : Le modèle n'a pas pu lier l'image au texte. Ceci est un problème connu (#2751) lié aux versions de Transformers/Gemma. Assurez-vous d'utiliser les versions spécifiées dans requirements.txt."
        else:
            return f"❌ Erreur lors de l'analyse avec Gemma 3n : {str(e)}"

def analyze_text_multilingual(text):
    """Analyse un texte avec le modèle Gemma 3n E4B IT."""
    if not st.session_state.model_loaded:
        if not restore_model_from_cache():
            st.warning("Modèle non chargé. Veuillez le charger via les réglages avant d'analyser.")
            return "❌ Modèle non chargé. Veuillez le charger dans les réglages."
        else:
            st.info("Modèle restauré depuis le cache pour l'analyse.")

    model, processor = st.session_state.model, st.session_state.processor
    if not model or not processor:
        return "❌ Modèle Gemma non disponible. Veuillez recharger le modèle."
    
    try:
        if st.session_state.language == "fr":
            prompt_template = f"Tu es un assistant agricole expert. Analyse ce problème de plante : \n\n**Description du problème :**\n{text}\n\n**Instructions :**\n1. **Diagnostic** : Quel est le problème principal ?\n2. **Causes** : Quelles sont les causes possibles ?\n3. **Traitement** : Quelles sont les actions à entreprendre ?\n4. **Prévention** : Comment éviter le problème à l'avenir ?"
        else:
            prompt_template = f"You are an expert agricultural assistant. Analyze this plant problem: \n\n**Problem Description:**\n{text}\n\n**Instructions:**\n1. **Diagnosis**: What is the main problem?\n2. **Causes**: What are the possible causes?\n3. **Treatment**: What actions should be taken?\n4. **Prevention**: How to avoid the problem in the future?"
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_template}]}]
        
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        device = getattr(model, 'device', 'cpu')
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                input_ids=inputs["input_ids"].to(device), # Passer explicitement input_ids
                attention_mask=inputs["attention_mask"].to(device), # Passer explicitement attention_mask
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            response = processor.decode(generation[0][input_len:], skip_special_tokens=True)
        
        cleaned_response = response.strip()
        cleaned_response = cleaned_response.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()

        return cleaned_response
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse de texte : {e}"

# --- Interface Principale ---

# Boutons de contrôle
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if st.button(t("toggle_mode"), type="secondary"):
        toggle_mobile_mode()
        st.rerun()
with col2:
    if st.button(t("settings_button"), type="secondary"):
        # Bascule la sidebar
        st.sidebar_state = not st.sidebar_state if hasattr(st, 'sidebar_state') else True
        st.rerun()

# Affichage conditionnel selon le mode
if is_mobile():
    # Mode Mobile
    st.markdown('<div class="mobile-container">', unsafe_allow_html=True)
    
    # Header mobile
    st.markdown(f'''
    <div class="mobile-header">
        <h1>{t("title")}</h1>
        <div class="mobile-status">{t("offline_status")} ✅</div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown(f'<p style="text-align: center; color: #666;">{t("mobile_desc")}</p>', unsafe_allow_html=True)
    
else:
    # Mode Desktop
    st.markdown('<div class="desktop-container">', unsafe_allow_html=True)
    st.title(t("title"))
    st.markdown(t("subtitle"))

# Initialisation des variables de session
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_status' not in st.session_state:
    st.session_state.model_status = "Non chargé"
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'global_model_cache' not in st.session_state:
    st.session_state.global_model_cache = {}
if 'model_persistence_check' not in st.session_state:
    st.session_state.model_persistence_check = False
if 'model_load_time' not in st.session_state:
    st.session_state.model_load_time = None

if not st.session_state.model_loaded:
    if restore_model_from_cache():
        st.success("🔄 Modèle restauré automatiquement depuis le cache au démarrage.")
    else:
        st.info("💡 Cliquez sur 'Charger le modèle' dans les réglages pour commencer.")

with st.sidebar:
    st.header("⚙️ " + t("config_title"))
    st.info("📋 Panneau des réglages ouvert")
    
    st.subheader("🌐 " + t("language_selection").split(":")[0])
    language_options = ["Français", "English"]
    if 'language' not in st.session_state:
        st.session_state.language = 'fr'
    current_lang_index = 0 if st.session_state.language == "fr" else 1
    
    language_choice = st.selectbox(
        t("language_selection"),
        language_options,
        index=current_lang_index,
        help=t("language_help")
    )
    if st.session_state.language != ("fr" if language_choice == "Français" else "en"):
        st.session_state.language = "fr" if language_choice == "Français" else "en"
        st.rerun()

    st.divider()

    st.subheader(t("hf_token_title"))
    hf_token_found = HfFolder.get_token() or os.environ.get("HF_TOKEN")
    if hf_token_found:
        st.success(t("hf_token_found"))
    else:
        st.warning(t("hf_token_not_found"))
    st.info(t("hf_token_info"))
    st.markdown(f"[{t('get_hf_token')}](https://huggingface.co/settings/tokens)")

    st.divider()

    st.header(t("model_title"))
    if st.session_state.model_loaded and check_model_persistence():
        st.success(f"{t('model_loaded')} ({st.session_state.model_status})")
        if st.session_state.model_load_time:
            load_time_str = time.strftime('%H:%M:%S', time.localtime(st.session_state.model_load_time))
            st.write(f"{t('load_time')}{load_time_str}")
        if hasattr(st.session_state.model, 'device'):
            st.write(f"{t('device_used')}`{st.session_state.model.device}`")
        
        col1_btn, col2_btn = st.columns(2)
        with col1_btn:
            if st.button(t("reload_model"), type="secondary"):
                st.session_state.model_loaded = False
                st.session_state.model = None
                st.session_state.processor = None
                st.session_state.global_model_cache.clear()
                st.session_state.model_persistence_check = False
                st.rerun()
        with col2_btn:
            if st.button(t("force_persistence"), type="secondary"):
                if force_model_persistence():
                    st.success(t("persistence_success"))
                else:
                    st.error(t("persistence_failed"))
                st.rerun()
    else:
        st.warning(t("model_not_loaded"))
        if st.button(t("load_model"), type="primary"):
            try:
                with st.spinner(t("loading_model")):
                    model, processor = load_model()
                    if model and processor:
                        # Mettre à jour les variables de session
                        st.session_state.model = model
                        st.session_state.processor = processor
                        st.session_state.model_loaded = True
                        st.session_state.model_status = "Chargé avec succès"
                        st.session_state.model_load_time = time.time()
                        st.session_state.model_persistence_check = True
                        st.success(t("model_loaded_success"))
                    else:
                        st.error(t("model_load_failed"))
                        st.session_state.model_loaded = False
                        st.session_state.model_status = "Échec du chargement"
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement : {str(e)}")
                st.session_state.model_loaded = False
                st.session_state.model_status = f"Erreur : {str(e)}"
            st.rerun()

    st.divider()
    st.subheader(t("persistence_title"))
    if st.session_state.model_loaded and st.session_state.model_persistence_check:
        st.success(t("persistence_loaded"))
    elif st.session_state.model_loaded:
        st.warning(t("persistence_warning"))
    else:
        st.warning(t("persistence_not_loaded"))


# --- Onglets Principaux ---
tab1, tab2, tab3, tab4 = st.tabs(t("tabs"))

with tab1:
    st.header(t("image_analysis_title"))
    st.markdown(t("image_analysis_desc"))
    
    capture_option = st.radio(
        t("choose_method"),
        [t("upload_image"), t("webcam_capture")],
        horizontal=True,
        key="image_capture_method"
    )
    
    uploaded_file = None
    captured_image = None
    
    if capture_option == t("upload_image"):
        uploaded_file = st.file_uploader(
            t("choose_image"),
            type=['png', 'jpg', 'jpeg'],
            help="Formats acceptés : PNG, JPG, JPEG (max 200MB). Privilégiez des images claires.",
            accept_multiple_files=False,
            key="image_uploader"
        )
        if uploaded_file is not None:
            MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024
            if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                st.error(t("file_too_large_error"))
                uploaded_file = None
            elif uploaded_file.size == 0:
                st.error(t("empty_file_error"))
                uploaded_file = None
            elif uploaded_file.size > (MAX_FILE_SIZE_BYTES * 0.8):
                st.warning(t("file_size_warning"))
    else:
        st.markdown(t("webcam_title"))
        st.info(t("webcam_info"))
        captured_image = st.camera_input(t("take_photo"), key="webcam_capture")
    
    image = None
    image_source = None
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_source = "upload"
        except Exception as e:
            st.error(t("image_processing_error"))
            st.info(t("try_different_image"))
    elif captured_image is not None:
        try:
            image = Image.open(captured_image)
            image_source = "webcam"
        except Exception as e:
            st.error(t("image_processing_error_webcam"))
            st.info(t("try_retake_photo"))
    
    if image is not None:
        try:
            original_size = image.size
            image, was_resized = resize_image_if_needed(image, max_size=(800, 800))
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption=f"Image ()" if image_source else "Image", use_container_width=True)
                if was_resized:
                    st.warning(t("image_resized_warning").format(new_size=image.size))
            
            with col2:
                st.markdown(t("image_info_title"))
                st.write(f"{t('format_label')}{image.format}")
                st.write(f"{t('original_size')}{original_size[0]}x{original_size[1]}{t('pixels')}")
                st.write(f"{t('current_size')}{image.size[0]}x{image.size[1]}{t('pixels')}")
                st.write(f"{t('mode_label')}{image.mode}")
            
            # Section de clarification de la culture
            st.markdown("---")
            st.subheader(t("culture_clarification"))
            
            culture_input = st.text_input(
                t("culture_question"),
                placeholder=t("culture_placeholder"),
                help=t("culture_help")
            )
            
            question = st.text_area(
                t("specific_question"),
                placeholder=t("question_placeholder"),
                height=100
            )
            
            if st.button(t("analyze_button"), disabled=not st.session_state.model_loaded, type="primary"):
                if not st.session_state.model_loaded:
                    st.error(t("model_not_loaded_error"))
                else:
                    with st.spinner(t("analyzing_image")):
                        # Construire le prompt avec la culture spécifiée
                        enhanced_prompt = ""
                        if culture_input:
                            enhanced_prompt += f"Culture spécifiée : {culture_input}. "
                        if question:
                            enhanced_prompt += f"Question : {question}. "
                        
                        result = analyze_image_multilingual(image, enhanced_prompt)
                    
                    st.markdown(t("analysis_results"))
                    
                    # Afficher la culture spécifiée si elle existe
                    if culture_input:
                        st.info(f"🌱 {t('culture_specified')} **{culture_input}**")
                    
                    st.markdown("---")
                    st.markdown(result)
                    
                    # Section d'export du diagnostic
                    st.markdown("---")
                    st.subheader(t("export_diagnostic"))
                    
                    # Préparer les informations pour l'export
                    timestamp = datetime.now().strftime("%d/%m/%Y à %H:%M")
                    image_info = {
                        "format": image.format if hasattr(image, 'format') else "N/A",
                        "size": f"{image.size[0]}x{image.size[1]} pixels" if hasattr(image, 'size') else "N/A"
                    } if image else None
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export HTML
                        html_content = generate_html_diagnostic(result, culture_input, image_info, timestamp)
                        st.download_button(
                            label=t("download_html"),
                            data=html_content,
                            file_name=t("html_filename").format(date=datetime.now().strftime("%Y%m%d_%H%M")),
                            mime="text/html",
                            help=t("export_html")
                        )
                    
                    with col2:
                        # Export Texte
                        text_content = generate_text_diagnostic(result, culture_input, image_info, timestamp)
                        st.download_button(
                            label=t("download_text"),
                            data=text_content,
                            file_name=t("text_filename").format(date=datetime.now().strftime("%Y%m%d_%H%M")),
                            mime="text/plain",
                            help=t("export_text")
                        )
        except Exception as e:
            st.error(t("image_processing_general_error"))

with tab2:
    st.header(t("text_analysis_title"))
    st.markdown(t("text_analysis_desc"))
    
    text_input = st.text_area(
        t("symptoms_desc"),
        placeholder=t("symptoms_placeholder"),
        height=150
    )
    
    if st.button("🧠 Analyser avec l'IA", disabled=not st.session_state.model_loaded, type="primary"):
        if not st.session_state.model_loaded:
            st.error("❌ Modèle non chargé. Veuillez le charger dans les réglages.")
        elif not text_input.strip():
            st.error("❌ Veuillez saisir une description des symptômes.")
        else:
            with st.spinner("🔍 Analyse de texte en cours..."):
                result = analyze_text_multilingual(text_input)
            
            st.markdown(t("analysis_results"))
            st.markdown("---")
            st.markdown(result)
            
            # Section d'export du diagnostic (analyse de texte)
            st.markdown("---")
            st.subheader(t("export_diagnostic"))
            
            # Préparer les informations pour l'export
            timestamp = datetime.now().strftime("%d/%m/%Y à %H:%M")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export HTML
                html_content = generate_html_diagnostic(result, None, None, timestamp)
                st.download_button(
                    label=t("download_html"),
                    data=html_content,
                    file_name=t("html_filename").format(date=datetime.now().strftime("%Y%m%d_%H%M")),
                    mime="text/html",
                    help=t("export_html")
                )
            
            with col2:
                # Export Texte
                text_content = generate_text_diagnostic(result, None, None, timestamp)
                st.download_button(
                    label=t("download_text"),
                    data=text_content,
                    file_name=t("text_filename").format(date=datetime.now().strftime("%Y%m%d_%H%M")),
                    mime="text/plain",
                    help=t("export_text")
                )

with tab3:
    st.header(t("manual_title"))
    
    manual_content = {
        "fr": """
        ## 🚀 **GUIDE DE DÉMARRAGE RAPIDE**
        
        ### **Étape 1 : Configuration Initiale**
        1. **Choisir la langue** : Dans la sidebar, sélectionnez Français ou English
        2. **Charger le modèle** : Cliquez sur 'Charger le modèle Gemma 3n E4B IT'
        3. **Attendre le chargement** : Le processus peut prendre 1-2 minutes
        4. **Vérifier le statut** : Le modèle doit afficher "✅ Chargé"

        ### **Étape 2 : Première Analyse**
        1. **Aller dans l'onglet "📸 Analyse d'Image"**
        2. **Télécharger une photo** de plante malade
        3. **Cliquer sur "🔬 Analyser avec l'IA"**
        4. **Consulter les résultats** avec recommandations

        ## 📱 **UTILISATION DU MODE MOBILE**
        
        ### **Activation du Mode Mobile**
        - Cliquer sur le bouton "🔄 Changer de mode" en haut de l'interface
        - L'interface se transforme en simulation d'application mobile
        - Statut "Mode: OFFLINE" visible pour les démonstrations

        ### **Avantages du Mode Mobile**
        - ✅ **Démonstration offline** : Parfait pour les présentations
        - ✅ **Interface intuitive** : Similaire aux vraies applications mobiles
        - ✅ **Accessibilité** : Fonctionne sur tous les appareils
        - ✅ **Performance** : Optimisé pour les ressources limitées

        ## 📸 **ANALYSE D'IMAGES**

        ### **Types d'Images Acceptées**
        - **Formats** : PNG, JPG, JPEG
        - **Taille maximale** : 200MB
        - **Qualité recommandée** : Images claires et bien éclairées

        ### **Bonnes Pratiques pour les Photos**
        1. **Éclairage** : Utiliser la lumière naturelle quand possible
        2. **Focus** : S'assurer que la zone malade est nette
        3. **Cadrage** : Inclure la plante entière et les zones affectées
        4. **Angles multiples** : Prendre plusieurs photos sous différents angles

        ### **Processus d'Analyse**
        1. **Téléchargement** : Glisser-déposer ou cliquer pour sélectionner
        2. **Préparation** : L'image est automatiquement redimensionnée si nécessaire
        3. **Analyse IA** : Le modèle Gemma 3n analyse l'image
        4. **Résultats** : Diagnostic détaillé avec recommandations

        ### **Interprétation des Résultats**
        Les résultats incluent :
        - 🎯 **Diagnostic probable** : Nom de la maladie identifiée
        - 🔍 **Symptômes observés** : Description détaillée des signes
        - 💡 **Causes possibles** : Facteurs environnementaux ou pathogènes
        - 💊 **Traitements recommandés** : Solutions pratiques
        - 🛡️ **Mesures préventives** : Conseils pour éviter la récurrence

        ## 💬 **ANALYSE DE TEXTE**

        ### **Quand Utiliser l'Analyse de Texte**
        - Pas de photo disponible
        - Symptômes difficiles à photographier
        - Besoin de conseils généraux
        - Vérification de diagnostic

        ### **Comment Décrire les Symptômes**
        **Informations importantes à inclure :**
        - 🌿 **Type de plante** : Nom de l'espèce si connu
        - 🎨 **Couleur des feuilles** : Vert, jaune, brun, noir, etc.
        - 🔍 **Forme des taches** : Circulaires, irrégulières, linéaires
        - 📍 **Localisation** : Feuilles, tiges, fruits, racines
        - ⏰ **Évolution** : Depuis quand, progression rapide ou lente
        - 🌍 **Conditions** : Humidité, température, saison

        ### **Exemple de Description Efficace**
        ```
        "Mes plants de tomates ont des taches brunes circulaires sur les feuilles inférieures. 
        Les taches ont un contour jaune et apparaissent depuis une semaine. 
        Il a beaucoup plu récemment et l'air est très humide. 
        Les taches s'étendent progressivement vers le haut de la plante."
        ```

        ## ⚙️ **CONFIGURATION ET PARAMÈTRES**

        ### **Paramètres de Langue**
        - **Français** : Interface et réponses en français
        - **English** : Interface and responses in English
        - **Changement** : Via la sidebar, effet immédiat

        ### **Gestion du Modèle IA**
        - **Chargement** : Bouton "Charger le modèle" dans la sidebar
        - **Statut** : Indicateur visuel du statut du modèle
        - **Rechargement** : Option pour recharger le modèle si nécessaire
        - **Persistance** : Le modèle reste en mémoire pour les analyses suivantes

        ### **Jeton Hugging Face (HF_TOKEN)**
        **Pourquoi l'utiliser ?**
        - Évite les erreurs d'accès (403)
        - Améliore la stabilité du téléchargement
        - Accès prioritaire aux modèles

        **Comment l'obtenir :**
        1. Aller sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        2. Créer un nouveau jeton avec les permissions "read"
        3. Copier le jeton généré
        4. Définir la variable d'environnement : `HF_TOKEN=votre_jeton`

        ## 🎯 **CAS D'USAGE PRATIQUES**

        ### **Scénario 1 : Diagnostic de Mildiou**
        1. **Symptômes** : Taches brunes sur feuilles de tomate
        2. **Photo** : Prendre une photo des feuilles affectées
        3. **Analyse** : L'IA identifie le mildiou précoce
        4. **Traitement** : Recommandations de fongicides et mesures préventives

        ### **Scénario 2 : Problème de Nutrition**
        1. **Symptômes** : Feuilles jaunies, croissance ralentie
        2. **Description** : Décrire les conditions de culture
        3. **Analyse** : L'IA suggère une carence en azote
        4. **Solution** : Recommandations d'engrais et d'amendements

        ## 🔧 **DÉPANNAGE**

        ### **Problèmes Courants**

        #### **Le modèle ne se charge pas**
        **Solutions :**
        - Vérifier la connexion internet
        - S'assurer d'avoir suffisamment de RAM (8GB minimum)
        - Redémarrer l'application
        - Vérifier le jeton HF_TOKEN

        #### **Erreur de mémoire**
        **Solutions :**
        - Fermer d'autres applications
        - Redémarrer l'ordinateur
        - Utiliser un modèle plus léger
        - Libérer de l'espace disque

        #### **Analyse trop lente**
        **Solutions :**
        - Réduire la taille des images
        - Utiliser des images de meilleure qualité
        - Vérifier la connexion internet
        - Patienter lors du premier chargement

        ### **Messages d'Erreur Courants**

        #### **"Erreur : Le fichier est trop volumineux"**
        - Réduire la taille de l'image (maximum 200MB)
        - Utiliser un format de compression (JPG au lieu de PNG)

        #### **"Modèle non chargé"**
        - Cliquer sur "Charger le modèle" dans la sidebar
        - Attendre la fin du chargement
        - Vérifier les messages d'erreur

        ## 🌍 **UTILISATION EN ZONES RURALES**

        ### **Avantages pour les Agriculteurs**
        - **Accessibilité** : Fonctionne sans internet constant
        - **Simplicité** : Interface intuitive
        - **Rapidité** : Diagnostic en quelques secondes
        - **Économique** : Gratuit et sans abonnement

        ### **Recommandations d'Usage**
        1. **Formation** : Former les utilisateurs aux bonnes pratiques
        2. **Validation** : Confirmer les diagnostics critiques avec des experts
        3. **Documentation** : Garder des traces des analyses
        4. **Suivi** : Utiliser l'application pour le suivi des traitements

        ## ⚠️ **AVERTISSEMENTS IMPORTANTS**

        ### **Limitations de l'IA**
        - Les résultats sont à titre indicatif uniquement
        - L'IA peut faire des erreurs de diagnostic
        - Les conditions locales peuvent affecter les recommandations
        - L'évolution des maladies peut être imprévisible

        ### **Responsabilité**
        - L'utilisateur reste responsable des décisions prises
        - Consulter un expert pour les cas critiques
        - Suivre les consignes de sécurité des produits
        - Adapter les traitements aux conditions locales

        ## 📞 **SUPPORT ET CONTACT**

        ### **Informations de Contact**
        - **Créateur** : Sidoine Kolaolé YEBADOKPO
        - **Localisation** : Bohicon, République du Bénin
        - **Téléphone** : +229 01 96 91 13 46
        - **Email** : syebadokpo@gmail.com
        - **LinkedIn** : linkedin.com/in/sidoineko
        - **Portfolio** : [Hugging Face Portfolio](https://huggingface.co/spaces/Sidoineko/portfolio)

        ### **Ressources Supplémentaires**
        - **Documentation technique** : README.md du projet
        - **Code source** : Disponible sur GitHub
        - **Démo en ligne** : Hugging Face Spaces
        - **Version compétition** : [Kaggle Notebook](https://www.kaggle.com/code/sidoineyebadokpo/agrilens-ai?scriptVersionId=253640926)

        ---

        *Manuel créé par Sidoine Kolaolé YEBADOKPO - Version 2.0 - Juillet 2025*
        """,
        "en": """
        ## 🚀 **QUICK START GUIDE**
        
        ### **Step 1: Initial Configuration**
        1. **Choose language** : In the sidebar, select Français or English
        2. **Load the model** : Click on 'Load Gemma 3n E4B IT Model'
        3. **Wait for loading** : The process may take 1-2 minutes
        4. **Check status** : The model should display "✅ Loaded"

        ### **Step 2: First Analysis**
        1. **Go to the "📸 Image Analysis" tab**
        2. **Upload a photo** of a diseased plant
        3. **Click on "🔬 Analyze with AI"**
        4. **Review the results** with recommendations

        ## 📱 **MOBILE MODE USAGE**
        
        ### **Activating Mobile Mode**
        - Click the "🔄 Toggle Mode" button at the top of the interface
        - The interface transforms into a mobile app simulation
        - "Mode: OFFLINE" status visible for demonstrations

        ### **Mobile Mode Advantages**
        - ✅ **Offline demonstration** : Perfect for presentations
        - ✅ **Intuitive interface** : Similar to real mobile applications
        - ✅ **Accessibility** : Works on all devices
        - ✅ **Performance** : Optimized for limited resources

        ## 📸 **IMAGE ANALYSIS**

        ### **Accepted Image Types**
        - **Formats** : PNG, JPG, JPEG
        - **Maximum size** : 200MB
        - **Recommended quality** : Clear and well-lit images

        ### **Best Practices for Photos**
        1. **Lighting** : Use natural light when possible
        2. **Focus** : Ensure the diseased area is sharp
        3. **Framing** : Include the entire plant and affected areas
        4. **Multiple angles** : Take several photos from different angles

        ### **Analysis Process**
        1. **Upload** : Drag and drop or click to select
        2. **Preparation** : Image is automatically resized if necessary
        3. **AI Analysis** : The Gemma 3n model analyzes the image
        4. **Results** : Detailed diagnosis with recommendations

        ### **Interpreting Results**
        Results include:
        - 🎯 **Probable diagnosis** : Name of the identified disease
        - 🔍 **Observed symptoms** : Detailed description of signs
        - 💡 **Possible causes** : Environmental factors or pathogens
        - 💊 **Recommended treatments** : Practical solutions
        - 🛡️ **Preventive measures** : Advice to prevent recurrence

        ## 💬 **TEXT ANALYSIS**

        ### **When to Use Text Analysis**
        - No photo available
        - Symptoms difficult to photograph
        - Need for general advice
        - Diagnosis verification

        ### **How to Describe Symptoms**
        **Important information to include:**
        - 🌿 **Plant type** : Species name if known
        - 🎨 **Leaf color** : Green, yellow, brown, black, etc.
        - 🔍 **Spot shape** : Circular, irregular, linear
        - 📍 **Location** : Leaves, stems, fruits, roots
        - ⏰ **Evolution** : Since when, rapid or slow progression
        - 🌍 **Conditions** : Humidity, temperature, season

        ### **Example of Effective Description**
        ```
        "My tomato plants have brown circular spots on the lower leaves. 
        The spots have a yellow border and appeared a week ago. 
        It has rained a lot recently and the air is very humid. 
        The spots are gradually spreading upward on the plant."
        ```

        ## ⚙️ **CONFIGURATION AND PARAMETERS**

        ### **Language Settings**
        - **Français** : Interface and responses in French
        - **English** : Interface and responses in English
        - **Change** : Via sidebar, immediate effect

        ### **AI Model Management**
        - **Loading** : "Load Model" button in sidebar
        - **Status** : Visual indicator of model status
        - **Reload** : Option to reload model if necessary
        - **Persistence** : Model remains in memory for subsequent analyses

        ### **Hugging Face Token (HF_TOKEN)**
        **Why use it?**
        - Avoids access errors (403)
        - Improves download stability
        - Priority access to models

        **How to get it:**
        1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        2. Create a new token with "read" permissions
        3. Copy the generated token
        4. Set environment variable: `HF_TOKEN=your_token`

        ## 🎯 **PRACTICAL USE CASES**

        ### **Scenario 1: Blight Diagnosis**
        1. **Symptoms** : Brown spots on tomato leaves
        2. **Photo** : Take a photo of affected leaves
        3. **Analysis** : AI identifies early blight
        4. **Treatment** : Fungicide recommendations and preventive measures

        ### **Scenario 2: Nutrition Problem**
        1. **Symptoms** : Yellowed leaves, slowed growth
        2. **Description** : Describe growing conditions
        3. **Analysis** : AI suggests nitrogen deficiency
        4. **Solution** : Fertilizer and amendment recommendations

        ## 🔧 **TROUBLESHOOTING**

        ### **Common Problems**

        #### **Model won't load**
        **Solutions:**
        - Check internet connection
        - Ensure sufficient RAM (8GB minimum)
        - Restart the application
        - Verify HF_TOKEN

        #### **Memory error**
        **Solutions:**
        - Close other applications
        - Restart computer
        - Use a lighter model
        - Free up disk space

        #### **Analysis too slow**
        **Solutions:**
        - Reduce image size
        - Use better quality images
        - Check internet connection
        - Be patient during first loading

        ### **Common Error Messages**

        #### **"Error: File too large"**
        - Reduce image size (maximum 200MB)
        - Use compression format (JPG instead of PNG)

        #### **"Model not loaded"**
        - Click "Load Model" in sidebar
        - Wait for loading to complete
        - Check error messages

        ## 🌍 **RURAL AREA USAGE**

        ### **Advantages for Farmers**
        - **Accessibility** : Works without constant internet
        - **Simplicity** : Intuitive interface
        - **Speed** : Diagnosis in seconds
        - **Economic** : Free and no subscription

        ### **Usage Recommendations**
        1. **Training** : Train users in best practices
        2. **Validation** : Confirm critical diagnoses with experts
        3. **Documentation** : Keep records of analyses
        4. **Follow-up** : Use application for treatment monitoring

        ## ⚠️ **IMPORTANT WARNINGS**

        ### **AI Limitations**
        - Results are for guidance only
        - AI can make diagnostic errors
        - Local conditions can affect recommendations
        - Disease evolution can be unpredictable

        ### **Responsibility**
        - User remains responsible for decisions made
        - Consult expert for critical cases
        - Follow product safety instructions
        - Adapt treatments to local conditions

        ## 📞 **SUPPORT AND CONTACT**

        ### **Contact Information**
        - **Creator** : Sidoine Kolaolé YEBADOKPO
        - **Location** : Bohicon, Benin Republic
        - **Phone** : +229 01 96 91 13 46
        - **Email** : syebadokpo@gmail.com
        - **LinkedIn** : linkedin.com/in/sidoineko
        - **Portfolio** : [Hugging Face Portfolio](https://huggingface.co/spaces/Sidoineko/portfolio)

        ### **Additional Resources**
        - **Technical documentation** : Project README.md
        - **Source code** : Available on GitHub
        - **Online demo** : Hugging Face Spaces
        - **Competition version** : [Kaggle Notebook](https://www.kaggle.com/code/sidoineyebadokpo/agrilens-ai?scriptVersionId=253640926)

        ---

        *Manual created by Sidoine Kolaolé YEBADOKPO - Version 2.0 - July 2025*
        """
    }
    st.markdown(manual_content[st.session_state.language])

with tab4:
    st.header(t("about_title"))
    
    st.markdown(t("mission_title"))
    st.markdown(t("mission_text"))
    
    st.markdown(t("features_title"))
    st.markdown(t("features_text"))
    
    st.markdown(t("technology_title"))
    
    is_local = os.path.exists(LOCAL_MODEL_PATH)
    
    if is_local:
        st.markdown(t("local_model_text").format(path=LOCAL_MODEL_PATH))
    else:
        st.markdown(t("online_model_text"))
    
    st.markdown(f"### {t('creator_title')}")
    st.markdown(f"{t('creator_name')}")
    st.markdown(f"📍 {t('creator_location')}")
    st.markdown(f"📞 {t('creator_phone')}")
    st.markdown(f"📧 {t('creator_email')}")
    st.markdown(f"🔗 [{t('creator_linkedin')}](https://{t('creator_linkedin')})")
    st.markdown(f"📁 {t('creator_portfolio')}")
    
    st.markdown(f"### {t('competition_title')}")
    st.markdown(t("competition_text"))
    
    st.markdown(t("warning_title"))
    st.markdown(t("warning_text"))
    
    st.markdown(t("support_title"))
    st.markdown(t("support_text"))

# --- Pied de page ---
st.markdown("---")
st.markdown(t("footer"))

# Fermer les divs selon le mode
if is_mobile():
    st.markdown('</div>', unsafe_allow_html=True)  # Fermer mobile-container
else:
    st.markdown('</div>', unsafe_allow_html=True)  # Fermer desktop-container