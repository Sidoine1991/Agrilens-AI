# --- IMPORTS ---
import streamlit as st
import os
import io
from PIL import Image
import torch
import gc
import time
import sys
import psutil
from datetime import datetime
from transformers import AutoProcessor, AutoModelForImageTextToText # Utilisation pour Gemma multimodal
from huggingface_hub import HfFolder, hf_hub_download, snapshot_download
from functools import lru_cache # Alternative pour le caching, mais st.cache_resource est mieux pour les modèles
import base64

# Charger les variables d'environnement depuis .env si le fichier existe
def load_env_file():
    """Charge les variables d'environnement depuis un fichier .env"""
    env_file = ".env"
    if os.path.exists(env_file):
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            print("✅ Variables d'environnement chargées depuis .env")
        except Exception as e:
            print(f"⚠️ Erreur chargement .env: {e}")

# Charger le fichier .env au démarrage
load_env_file()

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="AgriLens AI - Diagnostic des Plantes",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CONFIGURATION OPTIMISÉE POUR PERFORMANCE ---
# Configuration du modèle local
LOCAL_MODEL_PATH = r"D:\Dev\model_gemma"  # Chemin vers le modèle local
MODEL_ID_HF = "google/gemma-3n-e2b-it"  # ID Hugging Face (modèle Gemma 3n multimodal)

# Configuration optimisée pour Hugging Face Spaces
HF_TIMEOUT = 600  # 10 minutes de timeout pour le téléchargement (augmenté pour modèle local)
HF_RETRY_ATTEMPTS = 3  # Nombre de tentatives de téléchargement

def is_huggingface_spaces():
    """Détecte si l'application tourne sur Hugging Face Spaces"""
    return os.environ.get("SPACE_ID") is not None or "huggingface" in os.environ.get("HOSTNAME", "").lower()

# --- TRADUCTIONS ---
TRANSLATIONS = {
    "title": {"fr": "AgriLens AI", "en": "AgriLens AI"},
    "subtitle": {"fr": "Votre assistant IA pour le diagnostic des maladies de plantes", "en": "Your AI Assistant for Plant Disease Diagnosis"},
    "config_title": {"fr": "Configuration", "en": "Configuration"},
    "load_model_button": {"fr": "Charger le Modèle IA", "en": "Load AI Model"},
    "model_status": {"fr": "Statut du Modèle IA :", "en": "AI Model Status:"},
    "not_loaded": {"fr": "Non chargé", "en": "Not loaded"},
    "loaded": {"fr": "✅ Chargé", "en": "✅ Loaded"},
    "error": {"fr": "❌ Erreur", "en": "❌ Error"},
    "tabs": {"fr": ["📸 Analyse d'Image", "💬 Analyse de Texte", "📖 Manuel", "ℹ️ À propos"], "en": ["📸 Image Analysis", "💬 Text Analysis", "📖 Manual", "ℹ️ About"]},
    "image_analysis_title": {"fr": "🔍 Diagnostic par Image", "en": "🔍 Image Diagnosis"},
    "image_analysis_desc": {"fr": "Téléchargez ou capturez une photo de plante malade pour obtenir un diagnostic", "en": "Upload or capture a photo of a diseased plant to get a diagnosis"},
    "choose_image": {"fr": "Choisissez une image...", "en": "Choose an image..."},
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
    "competition_title": {"fr": "🏆 Version Compétition Kaggle", "en": "🏆 Kaggle Competition Version"},
    "competition_text": {"fr": "Cette première version d'AgriLens AI a été développée spécifiquement pour participer à la compétition Kaggle. Elle représente notre première production publique et démontre notre expertise en IA appliquée à l'agriculture.", "en": "This first version of AgriLens AI was specifically developed to participate in the Kaggle competition. It represents our first public production and demonstrates our expertise in AI applied to agriculture."},
    "footer": {"fr": "*AgriLens AI - Diagnostic intelligent des plantes avec IA*", "en": "*AgriLens AI - Intelligent plant diagnosis with AI*"},
    "export_button": {"fr": "Exporter les Résultats", "en": "Export Results"},
    "export_format": {"fr": "Format d'Exportation", "en": "Export Format"},
    "export_format_options": {"fr": ["JSON", "Markdown", "CSV"], "en": ["JSON", "Markdown", "CSV"]},
    "export_success": {"fr": "Résultats exportés avec succès !", "en": "Results exported successfully!"},
    "export_error": {"fr": "❌ Erreur lors de l'export", "en": "❌ Export error"},
    "html_filename": {"fr": "diagnostic_agrilens_{date}.html", "en": "agrilens_diagnosis_{date}.html"},
    "text_filename": {"fr": "diagnostic_agrilens_{date}.txt", "en": "agrilens_diagnosis_{date}.txt"},
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
    "webcam_info": {"fr": "Utilisez votre webcam pour capturer une image de votre plante.", "en": "Use your webcam to capture an image of your plant."},
    "take_photo": {"fr": "Prendre une photo", "en": "Take a photo"},
    "file_too_large": {"fr": "❌ Fichier trop volumineux. Taille maximale : 10 MB", "en": "❌ File too large. Maximum size: 10 MB"},
    "file_empty": {"fr": "❌ Fichier vide ou corrompu", "en": "❌ Empty or corrupted file"},
    "large_file_warning": {"fr": "⚠️ Fichier volumineux détecté. Le traitement peut prendre plus de temps.", "en": "⚠️ Large file detected. Processing may take longer."},
    "image_processing_error": {"fr": "❌ Erreur lors du traitement de l'image : ", "en": "❌ Error processing image: "},
    "image_processing_error_webcam": {"fr": "❌ Erreur lors du traitement de l'image webcam : ", "en": "❌ Error processing webcam image: "},
    "try_different_image": {"fr": "💡 Essayez avec une image différente ou vérifiez le format.", "en": "💡 Try with a different image or check the format."},
    "try_retake_photo": {"fr": "💡 Essayez de reprendre la photo ou utilisez l'upload.", "en": "💡 Try retaking the photo or use upload."},
    "ram_available": {"fr": "💾 RAM disponible : {ram} GB", "en": "💾 Available RAM: {ram} GB"},
    "ram_low_warning": {"fr": "⚠️ RAM faible détectée. Le modèle peut ne pas se charger.", "en": "⚠️ Low RAM detected. Model may not load."},
    "ram_check_error": {"fr": "❌ Erreur lors de la vérification de la RAM", "en": "❌ Error checking RAM"},
    "model_loading_attempt": {"fr": "🔄 Tentative de chargement du modèle : {model}", "en": "🔄 Attempting to load model: {model}"},
    "model_config": {"fr": "⚙️ Configuration : Device={device}, Dtype={dtype}, Quant={quant}", "en": "⚙️ Configuration: Device={device}, Dtype={dtype}, Quant={quant}"},
    "hf_token_configured": {"fr": "✅ Jeton HF configuré", "en": "✅ HF token configured"},
    "no_hf_token": {"fr": "⚠️ Aucun jeton HF trouvé", "en": "⚠️ No HF token found"},
    "quantization_4bit": {"fr": "🔧 Application de la quantification 4-bit", "en": "🔧 Applying 4-bit quantization"},
    "quantization_8bit": {"fr": "🔧 Application de la quantification 8-bit", "en": "🔧 Applying 8-bit quantization"},
    "bitsandbytes_no_gpu": {"fr": "⚠️ BitsAndBytes non disponible, pas de quantification", "en": "⚠️ BitsAndBytes not available, no quantization"},
    "bitsandbytes_error": {"fr": "❌ Erreur BitsAndBytes : {error}", "en": "❌ BitsAndBytes error: {error}"},
    "loading_processor": {"fr": "🔄 Chargement du processeur...", "en": "🔄 Loading processor..."},
    "processor_loaded": {"fr": "✅ Processeur chargé avec succès", "en": "✅ Processor loaded successfully"},
    "processor_load_error": {"fr": "❌ Erreur de chargement du processeur : {error}", "en": "❌ Processor load error: {error}"},
    "model_loaded": {"fr": "✅ Modèle chargé avec succès", "en": "✅ Model loaded successfully"},
    "model_load_error": {"fr": "❌ Erreur de chargement du modèle : {error}", "en": "❌ Model load error: {error}"},
    "model_loaded_device": {"fr": "✅ Modèle {model} chargé sur {device}", "en": "✅ Model {model} loaded on {device}"},
    "dependency_error": {"fr": "❌ Erreur de dépendance : {error}", "en": "❌ Dependency error: {error}"},
    "dependency_help": {"fr": "💡 Vérifiez que toutes les dépendances sont installées", "en": "💡 Check that all dependencies are installed"},
    "hf_access_error": {"fr": "❌ Erreur d'accès Hugging Face", "en": "❌ Hugging Face access error"},
    "hf_token_help": {"fr": "💡 Configurez votre jeton HF avec HF_TOKEN", "en": "💡 Configure your HF token with HF_TOKEN"},
    "model_config_error": {"fr": "❌ Erreur de configuration du modèle : {error}", "en": "❌ Model configuration error: {error}"},
    "unexpected_error": {"fr": "❌ Erreur inattendue : {error}", "en": "❌ Unexpected error: {error}"},
    "check_logs_help": {"fr": "💡 Vérifiez les logs pour plus de détails", "en": "💡 Check logs for more details"},
    "local_model_valid": {"fr": "✅ Modèle local valide trouvé : {path}", "en": "✅ Valid local model found: {path}"},
    "local_mode": {"fr": "🔄 Mode local activé", "en": "🔄 Local mode activated"},
    "local_model_unavailable": {"fr": "⚠️ Modèle local non disponible : {path}", "en": "⚠️ Local model unavailable: {path}"},
    "hf_mode": {"fr": "🌐 Mode Hugging Face activé : {model}", "en": "🌐 Hugging Face mode activated: {model}"},
    "gpu_memory": {"fr": "🎮 GPU détecté avec {memory} GB de mémoire", "en": "🎮 GPU detected with {memory} GB memory"},
    "gpu_memory_limited": {"fr": "⚠️ Mémoire GPU limitée", "en": "⚠️ Limited GPU memory"},
    "gpu_detection_error": {"fr": "❌ Erreur de détection GPU : {error}", "en": "❌ GPU detection error: {error}"},
    "cpu_mode": {"fr": "💻 Mode CPU activé", "en": "💻 CPU mode activated"},
    "loading_strategies": {"fr": "🔄 {count} stratégies de chargement disponibles", "en": "🔄 {count} loading strategies available"},
    "strategy_attempt": {"fr": "🔄 Tentative {current}/{total} : {name}", "en": "🔄 Attempt {current}/{total}: {name}"},
    "strategy_success": {"fr": "✅ Stratégie {name} réussie", "en": "✅ Strategy {name} successful"},
    "strategy_failed": {"fr": "❌ Stratégie {name} échouée : {error}", "en": "❌ Strategy {name} failed: {error}"},
    "strategy_config": {"fr": "⚙️ Configuration : {config}", "en": "⚙️ Configuration: {config}"},
    "all_strategies_failed": {"fr": "❌ Toutes les stratégies ont échoué", "en": "❌ All strategies failed"},
    "check_requirements": {"fr": "💡 Vérifiez les exigences système", "en": "💡 Check system requirements"},
    "check_local_model": {"fr": "💡 Vérifiez le modèle local", "en": "💡 Check local model"},
    "check_memory": {"fr": "💡 Vérifiez la mémoire disponible", "en": "💡 Check available memory"},
    "check_dependencies": {"fr": "💡 Vérifiez les dépendances", "en": "💡 Check dependencies"},
    "image_analysis_info": {"fr": "📸 Image : Format={format}, Taille={size}, Mode={mode}", "en": "📸 Image: Format={format}, Size={size}, Mode={mode}"},
    "image_rgb_converted": {"fr": "🔄 Image convertie en RGB (mode={mode})", "en": "🔄 Image converted to RGB (mode={mode})"},
    "image_resized": {"fr": "📏 Image redimensionnée", "en": "📏 Image resized"},
    "image_ready": {"fr": "✅ Image prête : Taille={size}, Mode={mode}", "en": "✅ Image ready: Size={size}, Mode={mode}"},
    "culture_considered": {"fr": "🌾 Culture considérée : {culture}", "en": "🌾 Culture considered: {culture}"},
    "location_considered": {"fr": "📍 Localisation considérée : {location}", "en": "📍 Location considered: {location}"},
    "agronomic_considered": {"fr": "🌱 Variables agronomiques : {vars}", "en": "🌱 Agronomic variables: {vars}"},
    "climatic_considered": {"fr": "🌤️ Variables climatiques : {vars}", "en": "🌤️ Climatic variables: {vars}"},
    "messages_structure": {"fr": "💬 Structure des messages : {count} messages, Type image={type}", "en": "💬 Message structure: {count} messages, Image type={type}"},
    "template_success": {"fr": "✅ Template appliqué avec succès", "en": "✅ Template applied successfully"},
    "fallback_prompt": {"fr": "🔄 Utilisation du prompt de secours", "en": "🔄 Using fallback prompt"},
    "generic_response_warning": {"fr": "⚠️ Réponse générique détectée. Essayez avec une image plus claire.", "en": "⚠️ Generic response detected. Try with a clearer image."},
    "image_resized_warning": {"fr": "⚠️ L'image a été redimensionnée à {new_size} pour optimiser le traitement.", "en": "⚠️ Image has been resized to {new_size} to optimize processing."},
    "image_info_title": {"fr": "**Informations de l'image :**", "en": "**Image information:**"},
    "format_label": {"fr": "Format : ", "en": "Format: "},
    "original_size": {"fr": "Taille originale : ", "en": "Original size: "},
    "current_size": {"fr": "Taille actuelle : ", "en": "Current size: "},
    "pixels": {"fr": " pixels", "en": " pixels"},
    "mode_label": {"fr": "Mode : ", "en": "Mode: "},
    "culture_clarification": {"fr": "🌱 Clarification de la Culture", "en": "🌱 Culture Clarification"},
    "culture_question": {"fr": "Quelle est la culture représentée dans cette image ?", "en": "What crop is represented in this image?"},
    "culture_placeholder": {"fr": "Ex: Tomate, Piment, Maïs, Haricot...", "en": "Ex: Tomato, Pepper, Corn, Bean..."},
    "culture_help": {"fr": "Spécifier la culture aide le modèle à se concentrer sur les maladies spécifiques à cette plante.", "en": "Specifying the crop helps the model focus on diseases specific to this plant."},
    "agronomic_variables_title": {"fr": "🌱 Variables Agronomiques", "en": "🌱 Agronomic Variables"},
    "agronomic_help": {"fr": "Ces informations aident à affiner le diagnostic en tenant compte des conditions de culture.", "en": "This information helps refine the diagnosis by considering growing conditions."},
    "soil_type": {"fr": "Type de sol", "en": "Soil type"},
    "soil_type_options": {"fr": ["Non spécifié", "Sableux", "Argileux", "Limoneux", "Humifère"], "en": ["Not specified", "Sandy", "Clay", "Loamy", "Humic"]},
    "soil_type_help": {"fr": "Type de sol dominant dans la zone de culture", "en": "Dominant soil type in the growing area"},
    "plant_age": {"fr": "Âge de la plante", "en": "Plant age"},
    "plant_age_placeholder": {"fr": "Ex: 2 mois, 45 jours...", "en": "Ex: 2 months, 45 days..."},
    "plant_age_help": {"fr": "Âge approximatif de la plante au moment de la photo", "en": "Approximate age of the plant when the photo was taken"},
    "planting_density": {"fr": "Densité de plantation", "en": "Planting density"},
    "planting_density_options": {"fr": ["Non spécifié", "Faible", "Moyenne", "Élevée"], "en": ["Not specified", "Low", "Medium", "High"]},
    "planting_density_help": {"fr": "Densité de plantation dans la parcelle", "en": "Planting density in the plot"},
    "irrigation": {"fr": "Irrigation", "en": "Irrigation"},
    "irrigation_options": {"fr": ["Non spécifié", "Pluie", "Goutte à goutte", "Aspersion", "Gravitaire"], "en": ["Not specified", "Rain", "Drip", "Sprinkler", "Gravity"]},
    "irrigation_help": {"fr": "Type d'irrigation utilisé", "en": "Type of irrigation used"},
    "fertilization": {"fr": "Fertilisation", "en": "Fertilization"},
    "fertilization_options": {"fr": ["Non spécifié", "Organique", "Chimique", "Mixte", "Aucune"], "en": ["Not specified", "Organic", "Chemical", "Mixed", "None"]},
    "fertilization_help": {"fr": "Type de fertilisation appliquée", "en": "Type of fertilization applied"},
    "crop_rotation": {"fr": "Rotation des cultures", "en": "Crop rotation"},
    "crop_rotation_options": {"fr": ["Non spécifié", "Oui", "Non", "Partielle"], "en": ["Not specified", "Yes", "No", "Partial"]},
    "crop_rotation_help": {"fr": "Pratique de rotation des cultures", "en": "Crop rotation practice"},
    "climatic_variables_title": {"fr": "🌤️ Variables Climatiques", "en": "🌤️ Climatic Variables"},
    "climatic_help": {"fr": "Ces informations aident à identifier les facteurs climatiques favorables aux maladies.", "en": "This information helps identify climatic factors favorable to diseases."},
    "temperature": {"fr": "Température", "en": "Temperature"},
    "temperature_placeholder": {"fr": "Ex: 25°C, 30-35°C...", "en": "Ex: 25°C, 30-35°C..."},
    "temperature_help": {"fr": "Température ambiante au moment de la photo", "en": "Ambient temperature when the photo was taken"},
    "humidity": {"fr": "Humidité", "en": "Humidity"},
    "humidity_options": {"fr": ["Non spécifié", "Faible (<40%)", "Moyenne (40-70%)", "Élevée (>70%)"], "en": ["Not specified", "Low (<40%)", "Medium (40-70%)", "High (>70%)"]},
    "humidity_help": {"fr": "Niveau d'humidité ambiante", "en": "Ambient humidity level"},
    "rainfall": {"fr": "Précipitations", "en": "Rainfall"},
    "rainfall_options": {"fr": ["Non spécifié", "Faible", "Modérée", "Abondante", "Sécheresse"], "en": ["Not specified", "Low", "Moderate", "Abundant", "Drought"]},
    "rainfall_help": {"fr": "Conditions de précipitations récentes", "en": "Recent precipitation conditions"},
    "season": {"fr": "Saison", "en": "Season"},
    "season_options": {"fr": ["Non spécifié", "Saison sèche", "Saison des pluies", "Transition"], "en": ["Not specified", "Dry season", "Rainy season", "Transition"]},
    "season_help": {"fr": "Saison actuelle", "en": "Current season"},
    "sun_exposure": {"fr": "Exposition solaire", "en": "Sun exposure"},
    "sun_exposure_options": {"fr": ["Non spécifié", "Pleine exposition", "Ombre partielle", "Ombre complète"], "en": ["Not specified", "Full exposure", "Partial shade", "Full shade"]},
    "sun_exposure_help": {"fr": "Niveau d'exposition solaire", "en": "Level of sun exposure"},
    "wind_conditions": {"fr": "Conditions de vent", "en": "Wind conditions"},
    "wind_conditions_options": {"fr": ["Non spécifié", "Calme", "Léger", "Modéré", "Fort"], "en": ["Not specified", "Calm", "Light", "Moderate", "Strong"]},
    "wind_conditions_help": {"fr": "Conditions de vent dans la zone", "en": "Wind conditions in the area"},
    "location_title": {"fr": "📍 Localisation", "en": "📍 Location"},
    "location_help": {"fr": "La localisation peut aider à identifier les maladies spécifiques à la région.", "en": "Location can help identify diseases specific to the region."},
    "location_method": {"fr": "Méthode de localisation", "en": "Location method"},
    "location_method_options": {"fr": ["GPS automatique", "Saisie manuelle"], "en": ["Automatic GPS", "Manual entry"]},
    "location_method_help": {"fr": "Choisissez comment spécifier la localisation", "en": "Choose how to specify the location"},
    "gps_info": {"fr": "📱 Sur mobile, la géolocalisation automatique sera utilisée si disponible.", "en": "📱 On mobile, automatic geolocation will be used if available."},
    "country": {"fr": "Pays", "en": "Country"},
    "country_placeholder": {"fr": "Ex: Bénin, France...", "en": "Ex: Benin, France..."},
    "city": {"fr": "Ville/Région", "en": "City/Region"},
    "city_placeholder": {"fr": "Ex: Bohicon, Cotonou...", "en": "Ex: Bohicon, Cotonou..."},
    "latitude": {"fr": "Latitude", "en": "Latitude"},
    "latitude_placeholder": {"fr": "Ex: 7.1761", "en": "Ex: 7.1761"},
    "longitude": {"fr": "Longitude", "en": "Longitude"},
    "longitude_placeholder": {"fr": "Ex: 2.0667", "en": "Ex: 2.0667"},
    "specific_question": {"fr": "Question spécifique (optionnel)", "en": "Specific question (optional)"},
    "question_placeholder": {"fr": "Ex: Est-ce grave ? Que faire en priorité ?", "en": "Ex: Is it serious? What to do first?"},
    "culture_specified": {"fr": "Culture spécifiée :", "en": "Specified crop:"},
    "location_display": {"fr": "📍 Localisation : {location}", "en": "📍 Location: {location}"},
    "export_diagnostic": {"fr": "📄 Exporter le Diagnostic", "en": "📄 Export Diagnosis"},
    "download_html": {"fr": "📄 Télécharger HTML", "en": "📄 Download HTML"},
    "download_text": {"fr": "📄 Télécharger Texte", "en": "📄 Download Text"},
    "export_html": {"fr": "Télécharger le diagnostic au format HTML avec mise en page", "en": "Download diagnosis in HTML format with layout"},
    "export_text": {"fr": "Télécharger le diagnostic au format texte simple", "en": "Download diagnosis in simple text format"},
    "symptoms_placeholder": {"fr": "Décrivez les symptômes observés sur votre plante...", "en": "Describe the symptoms observed on your plant..."},
    "symptoms_required": {"fr": "❌ Veuillez décrire les symptômes pour l'analyse.", "en": "❌ Please describe the symptoms for analysis."},
    "model_not_loaded_error": {"fr": "❌ Le modèle IA n'est pas chargé. Veuillez le charger dans les réglages.", "en": "❌ The AI model is not loaded. Please load it in the settings."},
    "image_processing_general_error": {"fr": "❌ Erreur générale lors du traitement de l'image : ", "en": "❌ General error processing image: "},
    "local_model_detected": {"fr": "🏠 Modèle local détecté - Stratégies optimisées", "en": "🏠 Local model detected - Optimized strategies"},
    "local_model_loading": {"fr": "🔄 Chargement du modèle local en cours...", "en": "🔄 Loading local model..."},
    "local_model_success": {"fr": "✅ Modèle local chargé avec succès !", "en": "✅ Local model loaded successfully!"},
    "local_model_failed": {"fr": "❌ Échec du chargement du modèle local", "en": "❌ Local model loading failed"},
    "analysis_started": {"fr": "🔊 Analyse démarrée - Notification sonore activée", "en": "🔊 Analysis started - Sound notification activated"},
    "analysis_completed": {"fr": "🔊 Analyse terminée - Notification sonore activée", "en": "🔊 Analysis completed - Sound notification activated"},
    "sound_notifications": {"fr": "🔊 Notifications Sonores", "en": "🔊 Sound Notifications"},
    "sound_enabled": {"fr": "✅ Notifications sonores activées", "en": "✅ Sound notifications enabled"},
    "sound_disabled": {"fr": "🔇 Notifications sonores désactivées", "en": "🔇 Sound notifications disabled"},
    "mission_title": {"fr": "### 🎯 Notre Mission", "en": "### 🎯 Our Mission"},
    "mission_text": {"fr": "AgriLens AI vise à démocratiser l'accès au diagnostic des maladies de plantes en utilisant l'intelligence artificielle. Notre objectif est d'aider les agriculteurs, les jardiniers et les professionnels de l'agriculture à identifier rapidement et précisément les problèmes de santé de leurs cultures.", "en": "AgriLens AI aims to democratize access to plant disease diagnosis using artificial intelligence. Our goal is to help farmers, gardeners, and agricultural professionals quickly and accurately identify health problems in their crops."},
    "features_title": {"fr": "### ✨ Fonctionnalités Principales", "en": "### ✨ Main Features"},
    "features_text": {"fr": "- **Diagnostic par Image** : Analysez des photos de plantes malades\n- **Analyse de Texte** : Décrivez les symptômes pour obtenir des conseils\n- **Support Multilingue** : Interface disponible en français et anglais\n- **Export de Résultats** : Sauvegardez vos diagnostics\n- **Interface Responsive** : Compatible mobile et desktop", "en": "- **Image Diagnosis** : Analyze photos of diseased plants\n- **Text Analysis** : Describe symptoms to get advice\n- **Multilingual Support** : Interface available in French and English\n- **Result Export** : Save your diagnoses\n- **Responsive Interface** : Mobile and desktop compatible"},
    "technology_title": {"fr": "### 🤖 Technologies Utilisées", "en": "### 🤖 Technologies Used"},
    "local_model_text": {"fr": "**Modèle Local** : Utilise un modèle Gemma 3n stocké localement à {path}", "en": "**Local Model** : Uses a Gemma 3n model stored locally at {path}"},
    "online_model_text": {"fr": "**Modèle en Ligne** : Utilise le modèle Gemma 3n via Hugging Face", "en": "**Online Model** : Uses the Gemma 3n model via Hugging Face"},
    "creator_portfolio": {"fr": "📁 [Portfolio Hugging Face](https://huggingface.co/spaces/Sidoineko/portfolio)", "en": "📁 [Hugging Face Portfolio](https://huggingface.co/spaces/Sidoineko/portfolio)"},
    "warning_title": {"fr": "### ⚠️ Avertissements Importants", "en": "### ⚠️ Important Warnings"},
    "warning_text": {"fr": "Les résultats fournis par l'IA sont à titre informatif uniquement. Pour des cas critiques, consultez toujours un expert en agriculture ou un phytopathologiste.", "en": "The results provided by AI are for informational purposes only. For critical cases, always consult an agricultural expert or plant pathologist."},
    "support_title": {"fr": "### 📞 Support et Contact", "en": "### 📞 Support and Contact"},
    "support_text": {"fr": "Pour toute question ou support technique, n'hésitez pas à nous contacter via les informations fournies ci-dessus.", "en": "For any questions or technical support, don't hesitate to contact us using the information provided above."}
}

def t(key):
    """Fonction de traduction simple."""
    if 'language' not in st.session_state:
        st.session_state.language = 'fr'
    lang = st.session_state.language
    return TRANSLATIONS.get(key, {}).get(lang, key)

def get_audio_html(audio_type="start"):
    """Génère le HTML pour les notifications sonores."""
    if audio_type == "start":
        # Son court pour le début de l'analyse
        audio_base64 = "data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
    else:
        # Son différent pour la fin de l'analyse
        audio_base64 = "data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
    
    return f"""
    <audio id="audio_{audio_type}" preload="auto">
        <source src="{audio_base64}" type="audio/wav">
    </audio>
    <script>
        function playAudio_{audio_type}() {{
            var audio = document.getElementById('audio_{audio_type}');
            audio.play();
        }}
    </script>
    """

def play_start_sound():
    """Joue le son de début d'analyse."""
    audio_html = get_audio_html("start")
    st.markdown(audio_html, unsafe_allow_html=True)
    st.markdown(
        """
        <script>
            setTimeout(function() {
                playAudio_start();
            }, 100);
        </script>
        """, 
        unsafe_allow_html=True
    )

def play_completion_sound():
    """Joue le son de fin d'analyse."""
    audio_html = get_audio_html("completion")
    st.markdown(audio_html, unsafe_allow_html=True)
    st.markdown(
        """
        <script>
            setTimeout(function() {
                playAudio_completion();
            }, 100);
        </script>
        """, 
        unsafe_allow_html=True
    )

# --- INITIALISATION DE LA LANGUE ET DES CONSTATATIONS GLOBALES ---
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

# --- MODE MOBILE DETECTION ---
def is_mobile():
    """Détecte si l'utilisateur est en mode mobile."""
    return st.session_state.get('mobile_mode', False)

def toggle_mobile_mode():
    """Bascule entre le mode desktop et mobile."""
    if 'mobile_mode' not in st.session_state:
        st.session_state.mobile_mode = False
    st.session_state.mobile_mode = not st.session_state.mobile_mode

# --- CSS POUR MODE MOBILE ---
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

# --- FONCTIONS UTILITAIRES SYSTÈME ---
def get_device():
    """Détermine le meilleur device disponible (GPU ou CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_model_path():
    """Détermine le chemin du modèle à utiliser (local ou Hugging Face)."""
    # Vérifier si le modèle local existe
    if os.path.exists(LOCAL_MODEL_PATH):
        # Vérifier que le dossier contient les fichiers nécessaires
        required_files = ['config.json', 'tokenizer.json']
        if all(os.path.exists(os.path.join(LOCAL_MODEL_PATH, f)) for f in required_files):
            return LOCAL_MODEL_PATH
    
    # Si le modèle local n'est pas disponible, utiliser Hugging Face
    return MODEL_ID_HF

def check_local_model():
    """Vérifie si le modèle local est valide et retourne un statut."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        return False, f"Dossier non trouvé : {LOCAL_MODEL_PATH}"
    
    try:
        files = os.listdir(LOCAL_MODEL_PATH)
        
        # Vérifier les fichiers requis
        required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
        missing_files = [f for f in required_files if f not in files]
        
        if missing_files:
            return False, f"Fichiers manquants : {', '.join(missing_files)}"
        
        # Vérifier qu'il y a des fichiers de poids du modèle
        model_files = [f for f in files if f.endswith(('.bin', '.safetensors', '.gguf'))]
        if not model_files:
            return False, "Aucun fichier de poids du modèle trouvé"
        
        # Calculer la taille totale
        total_size = sum(os.path.getsize(os.path.join(LOCAL_MODEL_PATH, f)) for f in files if os.path.isfile(os.path.join(LOCAL_MODEL_PATH, f)))
        size_gb = total_size / (1024**3)
        
        return True, f"Modèle valide ({len(files)} fichiers, {size_gb:.1f} GB)"
        
    except Exception as e:
        return False, f"Erreur lors de la vérification : {e}"

def diagnose_loading_issues():
    """Diagnostique les problèmes potentiels de chargement."""
    issues = []
    
    # Vérification des bibliothèques
    try:
        import transformers; issues.append(f"✅ Transformers v{transformers.__version__}")
        import torch; issues.append(f"✅ PyTorch v{torch.__version__}")
        if torch.cuda.is_available(): issues.append(f"✅ CUDA disponible : {torch.cuda.get_device_name(0)}")
        else: issues.append("⚠️ CUDA non disponible - utilisation CPU (plus lent)")
    except ImportError as e: issues.append(f"❌ Dépendance manquante : {e}")

    # Vérification du jeton Hugging Face
    hf_token = HfFolder.get_token() or os.environ.get("HF_TOKEN")
    if hf_token:
        issues.append("✅ Jeton Hugging Face configuré.")
    else:
        issues.append("⚠️ Jeton Hugging Face non configuré. Le téléchargement du modèle pourrait échouer.")

    # Vérification des ressources système
    try:
        mem = psutil.virtual_memory()
        issues.append(f"💾 RAM disponible : {mem.available // (1024**3)} GB")
        if mem.available < 4 * 1024**3: # Seuil de 4GB RAM
            issues.append("⚠️ RAM insuffisante (< 4GB). Le chargement risque d'échouer.")
    except ImportError: issues.append("⚠️ Impossible de vérifier la mémoire système.")
    
    return issues

def resize_image_if_needed(image, max_size=(800, 800)):
    """Redimensionne une image PIL si elle dépasse `max_size` tout en conservant les proportions et le format."""
    width, height = image.size
    if width > max_size[0] or height > max_size[1]:
        ratio = min(max_size[0] / width, max_size[1] / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Préserver le format de l'image originale
        if hasattr(image, 'format') and image.format:
            resized_image.format = image.format
        
        return resized_image, True
    return image, False

def afficher_ram_disponible():
    """Affiche l'utilisation de la RAM."""
    try:
        mem = psutil.virtual_memory()
        st.info(t("ram_available").format(ram=mem.available // (1024**3)))
        if mem.available < 4 * 1024**3:
            st.warning(t("ram_low_warning"))
    except ImportError:
        st.warning(t("ram_check_error"))

def generate_html_diagnostic(diagnostic_text, culture=None, image_info=None, timestamp=None, location=None):
    """
    Génère un fichier HTML formaté pour le diagnostic.
    
    Args:
        diagnostic_text (str): Le texte du diagnostic
        culture (str): La culture spécifiée
        image_info (dict): Informations sur l'image
        timestamp (str): Horodatage de l'analyse
        location (str): La localisation spécifiée
        
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
            {f'<p><strong>Localisation :</strong> {location}</p>' if location else ''}
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

def generate_text_diagnostic(diagnostic_text, culture=None, image_info=None, timestamp=None, location=None):
    """
    Génère un fichier texte formaté pour le diagnostic.
    
    Args:
        diagnostic_text (str): Le texte du diagnostic
        culture (str): La culture spécifiée
        image_info (dict): Informations sur l'image
        timestamp (str): Horodatage de l'analyse
        location (str): La localisation spécifiée
        
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
{f'Localisation : {location}' if location else ''}
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

# --- CHARGEMENT DU MODÈLE AVEC CACHING ---
# Modèle principal à utiliser (défini plus haut)
# Chemin local pour le modèle (optionnel, pour tester hors ligne)
# LOCAL_MODEL_PATH = "D:/Dev/model_gemma" # Décommentez et ajustez si vous avez un modèle local

@st.cache_resource(show_spinner=True) # Cache la ressource (modèle) entre les exécutions
def load_ai_model(model_identifier, device_map="auto", torch_dtype=torch.float16, quantization=None):
    """
    Charge le modèle et son tokenizer avec les configurations spécifiées.
    Retourne le modèle et le tokenizer, ou lève une exception en cas d'échec.
    """
    try:
        # Import local pour éviter les problèmes de scope
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        st.info(t("model_loading_attempt").format(model=model_identifier))
        st.info(t("model_config").format(device=device_map, dtype=torch_dtype, quant=quantization))
        
        # --- Configuration des arguments pour le chargement ---
        common_args = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True, # Aide à réduire l'utilisation CPU lors du chargement
            "device_map": device_map,
            "torch_dtype": torch_dtype,
        }
        
        # Configuration spéciale pour Gemma
        if "gemma" in model_identifier.lower():
            common_args.update({
                "use_cache": True,
            })
            
            # Flash attention seulement si disponible et sur GPU
            if device_map != "cpu" and torch.cuda.is_available():
                try:
                    import flash_attn
                    common_args["attn_implementation"] = "flash_attention_2"
                except ImportError:
                    pass  # Flash attention non disponible, utiliser l'attention standard
        
        # Ajouter le token seulement si c'est un modèle Hugging Face (pas local)
        if model_identifier.startswith("google/") or "/" in model_identifier:
            token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
            if token:
                common_args["token"] = token
                st.info(t("hf_token_configured"))
            else:
                st.warning(t("no_hf_token"))
        
        # Configuration de la quantisation (pour réduire l'empreinte mémoire)
        if quantization == "4bit":
            try:
                import bitsandbytes as bnb
                if bnb.cuda_setup.get_compute_capability() is not None:
                    common_args.update({
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": torch.float16,
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_quant_type": "nf4"
                    })
                    st.info(t("quantization_4bit"))
                else:
                    st.warning(t("bitsandbytes_no_gpu"))
            except Exception as e:
                st.warning(t("bitsandbytes_error").format(error=e))
        elif quantization == "8bit":
            try:
                import bitsandbytes as bnb
                if bnb.cuda_setup.get_compute_capability() is not None:
                    common_args.update({"load_in_8bit": True})
                    st.info(t("quantization_8bit"))
                else:
                    st.warning(t("bitsandbytes_no_gpu"))
            except Exception as e:
                st.warning(t("bitsandbytes_error").format(error=e))
        
        # --- Chargement du processor avec retry logic ---
        st.info(t("loading_processor"))
        processor = None
        for attempt in range(HF_RETRY_ATTEMPTS):
            try:
                # Préparer les arguments pour le processor
                processor_args = common_args.copy()
                # Ajouter timeout seulement pour les modèles Hugging Face (pas locaux)
                if model_identifier.startswith("google/") or "/" in model_identifier:
                    processor_args["timeout"] = HF_TIMEOUT
                
                processor = AutoProcessor.from_pretrained(
                    model_identifier, 
                    **processor_args
                )
                st.success(t("processor_loaded"))
                break
            except Exception as e:
                if attempt < HF_RETRY_ATTEMPTS - 1:
                    st.warning(f"Tentative {attempt + 1}/{HF_RETRY_ATTEMPTS} échouée pour le processor. Nouvelle tentative...")
                    time.sleep(5)  # Attendre 5 secondes avant de réessayer
                else:
                    st.error(t("processor_load_error").format(error=e))
                    raise
        
        # --- Chargement du modèle multimodal avec retry logic ---
        st.info(t("loading_model"))
        model = None
        for attempt in range(HF_RETRY_ATTEMPTS):
            try:
                # Préparer les arguments pour le modèle
                model_args = common_args.copy()
                # Ne pas ajouter timeout pour le modèle (seulement pour le processor)
                # Le timeout n'est pas supporté par AutoModelForImageTextToText
                
                # Détecter le type de modèle et utiliser la classe appropriée
                if "gemma" in model_identifier.lower():
                    # Pour les modèles Gemma, utiliser AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        model_identifier, 
                        **model_args
                    )
                else:
                    # Pour les autres modèles multimodaux, utiliser AutoModelForImageTextToText
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_identifier, 
                        **model_args
                    )
                st.success(t("model_loaded"))
                break
            except Exception as e:
                if attempt < HF_RETRY_ATTEMPTS - 1:
                    st.warning(f"Tentative {attempt + 1}/{HF_RETRY_ATTEMPTS} échouée pour le modèle. Nouvelle tentative...")
                    time.sleep(10)  # Attendre 10 secondes avant de réessayer
                else:
                    st.error(t("model_load_error").format(error=e))
                    raise
        
        st.success(t("model_loaded_device").format(model=model_identifier, device=device_map))
        return model, processor

    except ImportError as e:
        st.error(t("dependency_error").format(error=e))
        st.error(t("dependency_help"))
        raise ImportError(f"Erreur de dépendance : {e}. Assurez-vous que `transformers`, `torch`, `accelerate`, et `bitsandbytes` sont installés.")
    except ValueError as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
            st.error(t("hf_access_error"))
            st.error(t("hf_token_help"))
            raise ValueError("❌ Erreur d'accès Hugging Face (403). Vérifiez votre jeton Hugging Face (HF_TOKEN). Il doit être défini et valide.")
        else:
            st.error(t("model_config_error").format(error=e))
            raise ValueError(f"Erreur de configuration du modèle : {e}")
    except Exception as e:
        error_msg = str(e)
        
        # Gestion spécifique des erreurs de compatibilité
        if "unexpected keyword argument" in error_msg:
            st.error("❌ Erreur de compatibilité des arguments")
            st.error(f"Le modèle ne reconnaît pas certains arguments : {error_msg}")
            st.info("💡 Essayez de redémarrer l'application ou de mettre à jour les bibliothèques")
        elif "403" in error_msg or "Forbidden" in error_msg:
            st.error("❌ Erreur d'accès Hugging Face (403)")
            st.error("Vérifiez votre jeton Hugging Face (HF_TOKEN)")
        elif "timeout" in error_msg.lower():
            st.error("❌ Erreur de timeout")
            st.error("Le téléchargement a pris trop de temps. Vérifiez votre connexion internet.")
        else:
            st.error(t("unexpected_error").format(error=error_msg))
            st.error(t("check_logs_help"))
        
        raise RuntimeError(f"Une erreur est survenue lors du chargement du modèle : {error_msg}")

def get_model_and_processor():
    """
    Stratégie de chargement du modèle multimodal Gemma 3n e2b it.
    Utilise le modèle local s'il est disponible, sinon télécharge depuis Hugging Face.
    """
    # --- Diagnostic initial ---
    issues = diagnose_loading_issues()
    with st.expander("📊 Diagnostic système", expanded=False):
        for issue in issues:
            st.markdown(issue)

    # --- Détection du mode de chargement ---
    model_path = get_model_path()
    is_valid, status_message = check_local_model()
    
    if is_valid:
        st.success(t("local_model_valid").format(path=LOCAL_MODEL_PATH))
        st.info(f"📁 {status_message}")
        st.info(t("local_mode"))
    else:
        st.warning(t("local_model_unavailable").format(path=LOCAL_MODEL_PATH))
        st.error(f"❌ {status_message}")
        st.info(t("hf_mode").format(model=MODEL_ID_HF))

    # --- Stratégies de chargement ---
    strategies = []
    device = get_device()
    
    # Configuration spéciale pour Hugging Face Spaces
    if is_huggingface_spaces():
        st.info("🚀 Mode Hugging Face Spaces détecté - Configuration optimisée")
        # Sur HF Spaces, utiliser CPU avec des timeouts plus longs
        strategies.append({"name": "HF Spaces (CPU float32)", "config": {"device_map": "cpu", "torch_dtype": torch.float32, "quantization": None}})
        strategies.append({"name": "HF Spaces (CPU bfloat16)", "config": {"device_map": "cpu", "torch_dtype": torch.bfloat16, "quantization": None}})
        strategies.append({"name": "HF Spaces (CPU float32 fallback)", "config": {"device_map": "cpu", "torch_dtype": torch.float32, "quantization": None}})
    else:
        # Vérifier si CUDA est disponible
        if torch.cuda.is_available() and device == "cuda":
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.info(t("gpu_memory").format(memory=gpu_memory_gb))
                
                # Stratégies GPU par ordre de consommation mémoire décroissante
                if gpu_memory_gb >= 12: # Idéal pour float16
                    strategies.append({"name": "GPU (float16)", "config": {"device_map": "auto", "torch_dtype": torch.float16, "quantization": None}})
                if gpu_memory_gb >= 10: # Peut fonctionner avec float16
                    strategies.append({"name": "GPU (float16)", "config": {"device_map": "auto", "torch_dtype": torch.float16, "quantization": None}})
                if gpu_memory_gb >= 8: # Recommandé pour 8-bit quant.
                    strategies.append({"name": "GPU (8-bit quant.)", "config": {"device_map": "auto", "torch_dtype": torch.float16, "quantization": "8bit"}})
                if gpu_memory_gb >= 6: # Minimum pour 4-bit quant.
                    strategies.append({"name": "GPU (4-bit quant.)", "config": {"device_map": "auto", "torch_dtype": torch.float16, "quantization": "4bit"}})
                
                # Si la mémoire est très limitée, proposer une stratégie CPU
                if gpu_memory_gb < 6:
                    st.warning(t("gpu_memory_limited"))
            except Exception as e:
                st.warning(t("gpu_detection_error").format(error=e))
                device = "cpu"
    
    # Si CUDA n'est pas disponible ou a échoué, utiliser CPU
    if not torch.cuda.is_available() or device == "cpu":
        st.info(t("cpu_mode"))
        
        # Si on utilise le modèle local, ajouter des stratégies optimisées
        if model_path == LOCAL_MODEL_PATH:
            st.info(t("local_model_detected"))
            # Stratégies optimisées pour le modèle local (plus de mémoire, plus de temps)
            strategies.append({"name": "Local Model (CPU float32)", "config": {"device_map": "cpu", "torch_dtype": torch.float32, "quantization": None}})
            strategies.append({"name": "Local Model (CPU bfloat16)", "config": {"device_map": "cpu", "torch_dtype": torch.bfloat16, "quantization": None}})
        else:
            # Stratégies CPU optimisées pour les performances (sans quantisation)
            strategies.append({"name": "CPU (float32)", "config": {"device_map": "cpu", "torch_dtype": torch.float32, "quantization": None}})
            strategies.append({"name": "CPU (bfloat16)", "config": {"device_map": "cpu", "torch_dtype": torch.bfloat16, "quantization": None}})
        
        # Stratégie de fallback ultra-stable
        strategies.append({"name": "CPU (float32 - fallback)", "config": {"device_map": "cpu", "torch_dtype": torch.float32, "quantization": None}})
    else:
        # Stratégies CPU de fallback (plus lentes, mais plus robustes sur peu de ressources)
        strategies.append({"name": "CPU (bfloat16)", "config": {"device_map": "cpu", "torch_dtype": torch.bfloat16, "quantization": None}})
        strategies.append({"name": "CPU (float32)", "config": {"device_map": "cpu", "torch_dtype": torch.float32, "quantization": None}}) # Plus stable si bfloat16 échoue
    
    # --- Tentative de chargement via les stratégies ---
    st.info(t("loading_strategies").format(count=len(strategies)))
    
    for i, strat in enumerate(strategies, 1):
        st.info(t("strategy_attempt").format(current=i, total=len(strategies), name=strat['name']))
        try:
            model, processor = load_ai_model(
                model_path,  # Utilise le chemin détecté automatiquement
                device_map=strat["config"]["device_map"],
                torch_dtype=strat["config"]["torch_dtype"],
                quantization=strat["config"]["quantization"]
            )
            if model and processor:
                st.success(t("strategy_success").format(name=strat['name']))
                return model, processor
        except Exception as e:
            error_msg = str(e)
            st.warning(t("strategy_failed").format(name=strat['name'], error=error_msg))
            
            # Log détaillé pour le debugging
            with st.expander(f"🔍 Détails de l'erreur - {strat['name']}", expanded=False):
                st.code(f"Erreur: {error_msg}")
                st.info(t("strategy_config").format(config=strat['config']))
            
            # Nettoyage mémoire avant de passer à la stratégie suivante
            gc.collect()
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
            time.sleep(1) # Petite pause pour éviter les conflits

    # Si toutes les stratégies ont échoué, afficher un diagnostic détaillé
    st.error(t("all_strategies_failed"))
    st.error(t("check_requirements"))
    st.error(t("check_local_model"))
    st.error(t("check_memory"))
    st.error(t("check_dependencies"))
    raise RuntimeError("Toutes les stratégies de chargement du modèle ont échoué.")

# --- FONCTIONS D'ANALYSE ---
def analyze_image_multilingual(image, prompt="", culture="", agronomic_vars="", climatic_vars="", location=""):
    """Analyse une image avec Gemma 3n e2b it multimodal pour un diagnostic précis."""
    model, processor = st.session_state.model, st.session_state.processor
    if not model or not processor:
        return "❌ Modèle IA non chargé. Veuillez charger le modèle dans les réglages."

    try:
        # Vérification que l'image est bien présente
        if image is None:
            return "❌ Erreur : Aucune image fournie pour l'analyse."
        
        # Log de débogage pour vérifier l'image
        st.info(t("image_analysis_info").format(format=image.format, size=image.size, mode=image.mode))
        
        # S'assurer que l'image est en mode RGB (requis pour les modèles)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            st.info(t("image_rgb_converted").format(mode=image.mode))
        
        # Redimensionner l'image si nécessaire (comme dans le notebook Kaggle)
        if image.size[0] > 224 or image.size[1] > 224:
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            st.info(t("image_resized"))
        
        st.info(t("image_ready").format(size=image.size, mode=image.mode))
        
        # Construire le prompt en tenant compte de la culture, des variables agronomiques/climatiques et de la localisation
        additional_info = ""
        if culture and culture.strip():
            additional_info += f"Culture spécifiée : {culture.strip()}. "
            st.info(t("culture_considered").format(culture=culture.strip()))
        
        if location and location.strip():
            additional_info += f"Localisation : {location.strip()}. "
            st.info(t("location_considered").format(location=location.strip()))
        
        if agronomic_vars and agronomic_vars.strip():
            additional_info += f"Variables agronomiques : {agronomic_vars.strip()}. "
            st.info(t("agronomic_considered").format(vars=agronomic_vars.strip()))
        
        if climatic_vars and climatic_vars.strip():
            additional_info += f"Variables climatiques : {climatic_vars.strip()}. "
            st.info(t("climatic_considered").format(vars=climatic_vars.strip()))
        
        # Déterminer les messages selon la langue
        if st.session_state.language == "fr":
            user_instruction = f"{additional_info}Analyse cette image de plante malade et fournis un diagnostic SUCCINCT et STRUCTURÉ. Question : {prompt}" if prompt else f"{additional_info}Analyse cette image de plante malade et fournis un diagnostic SUCCINCT et STRUCTURÉ."
            system_message = f"""Tu es un expert en phytopathologie et agronomie. Analyse l'image fournie et fournis un diagnostic STRUCTURÉ et SUCCINCT en 3 parties obligatoires :

1. SYMPTÔMES VISIBLES : Description concise des signes visibles sur la plante (taches, déformations, jaunissement, etc.)

2. NOM DE LA MALADIE : Nom de la maladie la plus probable avec niveau de confiance (ex: "Mildiou du manioc (85%)")

3. TRAITEMENT RECOMMANDÉ : Actions concrètes à mettre en œuvre pour traiter la maladie (fongicide, amélioration ventilation, etc.)

{additional_info}

IMPORTANT : 
- Analyse UNIQUEMENT l'image fournie
- Sois précis et concis
- Inclus TOUJOURS les 3 sections
- Ne donne PAS de réponses génériques
- Prends en compte les variables agronomiques et climatiques fournies pour affiner le diagnostic"""
        else: # English
            user_instruction = f"{additional_info}Analyze this diseased plant image and provide a SUCCINCT and STRUCTURED diagnosis. Question: {prompt}" if prompt else f"{additional_info}Analyze this diseased plant image and provide a SUCCINCT and STRUCTURED diagnosis."
            system_message = f"""You are a plant pathology and agronomy expert. Analyze the provided image and provide a STRUCTURED and SUCCINCT diagnosis in 3 mandatory parts:

1. VISIBLE SYMPTOMS: Concise description of visible signs on the plant (spots, deformations, yellowing, etc.)

2. DISEASE NAME: Most likely disease name with confidence level (e.g., "Cassava Mosaic Virus (85%)")

3. RECOMMENDED TREATMENT: Concrete actions to implement to treat the disease (fungicide, ventilation improvement, etc.)

{additional_info}

IMPORTANT:
- Analyze ONLY the provided image
- Be precise and concise
- ALWAYS include the 3 sections
- Do NOT give generic responses
- Take into account the provided agronomic and climatic variables to refine the diagnosis"""
        
        # Structure des messages pour Gemma 3n e2b it multimodal (comme dans le notebook Kaggle)
        # Format simplifié sans message système séparé
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # Passer directement l'objet PIL Image
                    {"type": "text", "text": f"{system_message}\n\n{user_instruction} IMPORTANT : Analyse uniquement ce que tu vois dans cette image spécifique. Ne donne pas de réponse générique."}
                ]
            }
        ]
        
        # Log pour debug
        st.info(t("messages_structure").format(count=len(messages), type=type(image)))
        
        # Utiliser processor.apply_chat_template pour convertir le format conversationnel en tenseurs
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            st.info(t("template_success"))
        except Exception as template_error:
            st.error(f"❌ Erreur avec apply_chat_template: {template_error}")
            # Fallback : essayer un format plus simple
            st.info(t("fallback_prompt"))
            simple_prompt = f"{system_message}\n\n{user_instruction}"
            inputs = processor(simple_prompt, image, return_tensors="pt", padding=True, truncation=True)
        
        device = getattr(model, 'device', 'cpu')
        # Déplacer les tenseurs sur le bon device
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            # Configuration de génération avec max_new_tokens à 500
            generation = model.generate(
                **inputs, # Déballer le dictionnaire des inputs
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=100,
                repetition_penalty=1.1,
                use_cache=True,
                num_beams=1
            )
            
            response = processor.batch_decode(generation[:, input_len:], skip_special_tokens=True)[0]

        final_response = response.strip()
        # Nettoyage des tokens de contrôle si présents
        final_response = final_response.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()
        
        # Vérification que la réponse n'est pas générique
        generic_indicators = [
            "sans l'image", "sans voir l'image", "basé sur des connaissances générales",
            "without the image", "without seeing the image", "based on general knowledge",
            "veuillez me fournir l'image", "please provide the image", "aucune image"
        ]
        
        is_generic = any(indicator.lower() in final_response.lower() for indicator in generic_indicators)
        
        if is_generic:
            st.warning(t("generic_response_warning"))
            # Ajouter une instruction pour forcer l'analyse de l'image
            final_response += "\n\n⚠️ **Note importante** : Cette réponse semble générique. Veuillez vérifier que l'image a été correctement uploadée et réessayer l'analyse."
        
        # Formatage de la réponse pour l'affichage
        if st.session_state.language == "fr":
            return f"## 🧠 **Analyse par Gemma 3n e2b it**\n\n{final_response}"
        else:
            return f"## 🧠 **Analysis by Gemma 3n e2b it**\n\n{final_response}"
            
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
            return "❌ Erreur 403 - Accès refusé. Vérifiez votre jeton Hugging Face (HF_TOKEN) et les quotas."
        elif "Number of images does not match number of special image tokens" in error_msg:
            return "❌ Erreur : Le modèle n'a pas pu associer l'image au texte. Ceci est un bug connu (#2751) lié aux versions de Transformers/Gemma. Essayez de mettre à jour vos bibliothèques (`transformers`, `torch`, `accelerate`)."
        else:
            return f"❌ Erreur lors de l'analyse d'image : {e}"

def analyze_text_multilingual(text):
    """Analyse un texte avec le modèle Gemma 3n e2b it multimodal."""
    model, processor = st.session_state.model, st.session_state.processor
    if not model or not processor:
        return "❌ Modèle IA non chargé. Veuillez charger le modèle dans les réglages."
        
    try:
        # Construction du prompt selon la langue
        if st.session_state.language == "fr":
            prompt_template = f"Tu es un expert en pathologie végétale. Analyse ce problème de plante de manière SUCCINCTE et STRUCTURÉE : \n\n**Description :**\n{text}\n\n**Réponds avec EXACTEMENT ces 3 sections :**\n1. **SYMPTÔMES** (description courte)\n2. **NOM DE LA MALADIE/PROBLÈME** (avec niveau de confiance %)\n3. **TRAITEMENT** (actions concrètes)\n\nSois précis et concis. Maximum 150 mots."
        else: # English
            prompt_template = f"You are a plant pathology expert. Analyze this plant problem in a SUCCINCT and STRUCTURED manner: \n\n**Description:**\n{text}\n\n**Respond with EXACTLY these 3 sections:**\n1. **SYMPTOMS** (brief description)\n2. **DISEASE/PROBLEM NAME** (with confidence level %)\n3. **TREATMENT** (concrete actions)\n\nBe precise and concise. Maximum 150 words."
        
        # Format correct pour l'analyse de texte avec AutoProcessor
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
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=100,
                repetition_penalty=1.1,
                use_cache=True,
                num_beams=1
            )
            
            response = processor.batch_decode(generation[:, input_len:], skip_special_tokens=True)[0]
        
        cleaned_response = response.strip()
        cleaned_response = cleaned_response.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()

        return cleaned_response
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse de texte : {e}"

# --- INTERFACE UTILISATEUR STREAMLIT ---

# --- INITIALISATION DES VARIABLES DE SESSION ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_status' not in st.session_state:
    st.session_state.model_status = t("not_loaded")

# --- SIDEBAR (RÉGLAGES) ---
with st.sidebar:
    st.header("⚙️ " + t("config_title"))
    
    # Sélection de la langue
    st.subheader("🌐 " + t("language_selection"))
    language_options = ["Français", "English"]
    current_lang_index = 0 if st.session_state.language == "fr" else 1
    language_choice = st.selectbox(
        t("language_selection"),
        language_options,
        index=current_lang_index,
        help=t("language_help")
    )
    if st.session_state.language != ("fr" if language_choice == "Français" else "en"):
        st.session_state.language = "fr" if language_choice == "Français" else "en"
        st.rerun() # Recharge l'application pour appliquer la langue

    st.divider()

    # Configuration des notifications sonores
    st.subheader(t("sound_notifications"))
    if 'sound_enabled' not in st.session_state:
        st.session_state.sound_enabled = True
    
    sound_toggle = st.toggle(
        "🔊 Activer les notifications sonores",
        value=st.session_state.sound_enabled,
        help="Joue un son au début et à la fin de l'analyse"
    )
    
    if sound_toggle != st.session_state.sound_enabled:
        st.session_state.sound_enabled = sound_toggle
        if sound_toggle:
            st.success(t("sound_enabled"))
        else:
            st.info(t("sound_disabled"))
    
    st.divider()

    # Configuration du jeton Hugging Face
    st.subheader(t("hf_token_title"))
    hf_token_found = HfFolder.get_token() or os.environ.get("HF_TOKEN")
    if hf_token_found:
        st.success(t("hf_token_found"))
    else:
        st.warning(t("hf_token_not_found"))
    st.info(t("hf_token_info"))
    st.markdown(f"[{t('get_hf_token')}](https://huggingface.co/settings/tokens)")

    st.divider()

    # Gestion du modèle IA
    st.header(t("model_title"))
    if st.session_state.model_loaded:
        st.success(f"{t('model_status')} {st.session_state.model_status}")
        if st.session_state.model and hasattr(st.session_state.model, 'device'):
            st.write(f"{t('device_used')}`{st.session_state.model.device}`")
        
        col1_btn, col2_btn = st.columns(2)
        with col1_btn:
            if st.button(t("reload_model"), type="secondary"):
                st.session_state.model = None
                st.session_state.processor = None
                st.session_state.model_loaded = False
                st.session_state.model_status = t("not_loaded")
                # Désactive le cache pour forcer le rechargement
                if 'load_ai_model' in st.cache_resource.__wrapped__.__wrapped__.__self__.__dict__:
                    st.cache_resource.__wrapped__.__wrapped__.__self__['load_ai_model'].clear()
                st.rerun()
        with col2_btn:
            # Bouton pour forcer la persistance via @st.cache_resource
            if st.button(t("force_persistence"), type="secondary"):
                st.cache_resource.clear() # Efface le cache pour forcer le rechargement et la ré-application du cache
                st.success(t("persistence_success"))
                st.rerun()
    else:
        st.warning(f"{t('model_status')} {st.session_state.model_status}")
        if st.button(t("load_model_button"), type="primary"):
            # Essaye de charger le modèle manuellement
            try:
                model, processor = get_model_and_processor()
                st.session_state.model = model
                st.session_state.processor = processor
                st.session_state.model_loaded = True
                st.session_state.model_status = t("loaded")
                st.success(t("model_loaded_success"))
            except Exception as e:
                st.session_state.model_status = f"{t('error')} : {e}"
                st.error(t("model_load_failed"))
            st.rerun()

# --- BOUTONS DE CONTRÔLE ---
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if st.button("🔄 Changer de mode", type="secondary"):
        toggle_mobile_mode()
        st.rerun()
with col2:
    if st.button("⚙️ Réglages", type="secondary"):
        # Bascule la sidebar
        st.sidebar_state = not st.sidebar_state if hasattr(st, 'sidebar_state') else True
        st.rerun()

# --- AFFICHAGE CONDITIONNEL SELON LE MODE ---
if is_mobile():
    # Mode Mobile
    st.markdown('<div class="mobile-container">', unsafe_allow_html=True)
    
    # Header mobile
    st.markdown(f'''
    <div class="mobile-header">
        <h1>{t("title")}</h1>
        <div class="mobile-status">Mode: OFFLINE ✅</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Logo mobile
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("logo_app/logo_agrilesai.png", width=150)
        except:
            # Si le logo n'est pas disponible, afficher un emoji et le titre
            st.markdown("## 🌱 AgriLens AI")
    
    st.markdown(f'<p style="text-align: center; color: #666;">Interface simulant l\'application mobile offline</p>', unsafe_allow_html=True)
    
else:
    # Mode Desktop
    st.markdown('<div class="desktop-container">', unsafe_allow_html=True)
    
    # Logo et titre
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("logo_app/logo_agrilesai.png", width=200)
        except:
            # Si le logo n'est pas disponible, afficher un emoji et le titre
            st.markdown("## 🌱 AgriLens AI")
    
    st.title(t("title"))
    st.markdown(t("subtitle"))

# --- ONGLET PRINCIPAUX ---
tab1, tab2, tab3, tab4 = st.tabs(t("tabs"))

# --- ONGLET 1: ANALYSE D'IMAGE ---
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
                st.error(t("file_too_large"))
                uploaded_file = None
            elif uploaded_file.size == 0:
                st.error(t("file_empty"))
                uploaded_file = None
            elif uploaded_file.size > (MAX_FILE_SIZE_BYTES * 0.8):
                st.warning(t("large_file_warning"))
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
            st.error(t("image_processing_error") + str(e))
            st.info(t("try_different_image"))
    elif captured_image is not None:
        try:
            image = Image.open(captured_image)
            image_source = "webcam"
        except Exception as e:
            st.error(t("image_processing_error_webcam") + str(e))
            st.info(t("try_retake_photo"))
    
    if image is not None:
        try:
            original_size = image.size
            image, was_resized = resize_image_if_needed(image, max_size=(800, 800))
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption=f"Image ({image_source})", use_container_width=True)
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
            
            # Section des variables agronomiques
            st.markdown("---")
            st.subheader(t("agronomic_variables_title"))
            st.info(t("agronomic_help"))
            
            col1, col2 = st.columns(2)
            with col1:
                soil_type = st.selectbox(
                    t("soil_type"),
                    t("soil_type_options"),
                    help=t("soil_type_help")
                )
                
                plant_age = st.text_input(
                    t("plant_age"),
                    placeholder=t("plant_age_placeholder"),
                    help=t("plant_age_help")
                )
                
                planting_density = st.selectbox(
                    t("planting_density"),
                    t("planting_density_options"),
                    help=t("planting_density_help")
                )
            
            with col2:
                irrigation = st.selectbox(
                    t("irrigation"),
                    t("irrigation_options"),
                    help=t("irrigation_help")
                )
                
                fertilization = st.selectbox(
                    t("fertilization"),
                    t("fertilization_options"),
                    help=t("fertilization_help")
                )
                
                crop_rotation = st.selectbox(
                    t("crop_rotation"),
                    t("crop_rotation_options"),
                    help=t("crop_rotation_help")
                )
            
            # Section des variables climatiques
            st.markdown("---")
            st.subheader(t("climatic_variables_title"))
            st.info(t("climatic_help"))
            
            col3, col4 = st.columns(2)
            with col3:
                temperature = st.text_input(
                    t("temperature"),
                    placeholder=t("temperature_placeholder"),
                    help=t("temperature_help")
                )
                
                humidity = st.selectbox(
                    t("humidity"),
                    t("humidity_options"),
                    help=t("humidity_help")
                )
                
                rainfall = st.selectbox(
                    t("rainfall"),
                    t("rainfall_options"),
                    help=t("rainfall_help")
                )
            
            with col4:
                season = st.selectbox(
                    t("season"),
                    t("season_options"),
                    help=t("season_help")
                )
                
                sun_exposure = st.selectbox(
                    t("sun_exposure"),
                    t("sun_exposure_options"),
                    help=t("sun_exposure_help")
                )
                
                wind_conditions = st.selectbox(
                    t("wind_conditions"),
                    t("wind_conditions_options"),
                    help=t("wind_conditions_help")
                )
            
            # Section de localisation
            st.markdown("---")
            st.subheader(t("location_title"))
            st.info(t("location_help"))
            
            location_method = st.radio(
                t("location_method"),
                t("location_method_options"),
                help=t("location_method_help")
            )
            
            if location_method == t("location_method_options")[0]:  # GPS automatique
                # Note: Streamlit n'a pas de widget GPS natif, on utilise une explication
                st.info(t("gps_info"))
                location_input = ""
            else:
                col5, col6 = st.columns(2)
                with col5:
                    country = st.text_input(
                        t("country"),
                        placeholder=t("country_placeholder"),
                        help="Pays où l'image a été prise"
                    )
                    
                    city = st.text_input(
                        t("city"),
                        placeholder=t("city_placeholder"),
                        help="Ville ou région où l'image a été prise"
                    )
                
                with col6:
                    latitude = st.text_input(
                        t("latitude"),
                        placeholder=t("latitude_placeholder"),
                        help="Coordonnée GPS latitude"
                    )
                    
                    longitude = st.text_input(
                        t("longitude"),
                        placeholder=t("longitude_placeholder"),
                        help="Coordonnée GPS longitude"
                    )
                
                # Construire la chaîne de localisation
                location_parts = []
                if country: location_parts.append(f"Pays: {country}")
                if city: location_parts.append(f"Ville: {city}")
                if latitude and longitude: location_parts.append(f"GPS: {latitude}, {longitude}")
                location_input = "; ".join(location_parts) if location_parts else ""
            
            question = st.text_area(
                t("specific_question"),
                placeholder=t("question_placeholder"),
                height=100
            )
            
            if st.button(t("analyze_button"), disabled=not st.session_state.model_loaded, type="primary"):
                if not st.session_state.model_loaded:
                    st.error(t("model_not_loaded_error"))
                else:
                    # Notification sonore de début d'analyse
                    if st.session_state.get('sound_enabled', True):
                        play_start_sound()
                        st.info(t("analysis_started"))
                    
                    # Créer des placeholders pour la progression
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    # Afficher la barre de progression initiale
                    progress_placeholder.progress(0)
                    status_placeholder.info("🔍 Préparation de l'analyse...")
                    
                    # Collecter les variables agronomiques
                    agronomic_vars = []
                    if soil_type: agronomic_vars.append(f"Sol: {soil_type}")
                    if plant_age: agronomic_vars.append(f"Âge: {plant_age}")
                    if planting_density: agronomic_vars.append(f"Densité: {planting_density}")
                    if irrigation: agronomic_vars.append(f"Irrigation: {irrigation}")
                    if fertilization: agronomic_vars.append(f"Fertilisation: {fertilization}")
                    if crop_rotation: agronomic_vars.append(f"Rotation: {crop_rotation}")
                    agronomic_vars_str = "; ".join(agronomic_vars) if agronomic_vars else ""
                    
                    # Collecter les variables climatiques
                    climatic_vars = []
                    if temperature: climatic_vars.append(f"Température: {temperature}°C")
                    if humidity: climatic_vars.append(f"Humidité: {humidity}")
                    if rainfall: climatic_vars.append(f"Précipitations: {rainfall}")
                    if season: climatic_vars.append(f"Saison: {season}")
                    if sun_exposure: climatic_vars.append(f"Exposition: {sun_exposure}")
                    if wind_conditions: climatic_vars.append(f"Vent: {wind_conditions}")
                    climatic_vars_str = "; ".join(climatic_vars) if climatic_vars else ""
                    
                    # Construire le prompt avec la culture spécifiée
                    enhanced_prompt = ""
                    if culture_input:
                        enhanced_prompt += f"Culture spécifiée : {culture_input}. "
                    if question:
                        enhanced_prompt += f"Question : {question}. "
                    
                    # Simuler la progression pendant l'analyse
                    import time
                    
                    # Étapes de progression
                    steps = [
                        (10, "Préparation de l'image..."),
                        (25, "Chargement du modèle..."),
                        (40, "Analyse des caractéristiques visuelles..."),
                        (60, "Identification des symptômes..."),
                        (80, "Génération du diagnostic..."),
                        (95, "Finalisation de la réponse..."),
                        (100, "Analyse terminée !")
                    ]
                    
                    # Effectuer l'analyse avec mise à jour de la progression
                    for progress, status in steps[:-1]:  # Toutes sauf la dernière
                        progress_placeholder.progress(progress / 100)
                        status_placeholder.info(f"🔍 {status}")
                        time.sleep(0.3)  # Petite pause pour voir la progression
                    
                    # Effectuer l'analyse réelle
                    result = analyze_image_multilingual(image, enhanced_prompt, culture_input, agronomic_vars_str, climatic_vars_str, location_input)
                    
                    # Finaliser la progression
                    progress_placeholder.progress(1.0)
                    status_placeholder.success("✅ Analyse terminée !")
                    
                    # Notification sonore de fin d'analyse
                    if st.session_state.get('sound_enabled', True):
                        play_completion_sound()
                        st.success(t("analysis_completed"))
                    
                    # Effacer les placeholders après un court délai
                    time.sleep(1)
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    
                    st.markdown(t("analysis_results"))
                    
                    # Afficher la culture spécifiée si elle existe
                    if culture_input:
                        st.info(f"🌱 {t('culture_specified')} **{culture_input}**")
                    
                    # Afficher la localisation si elle existe
                    if location_input:
                        st.info(t("location_display").format(location=location_input))
                    
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
                        html_content = generate_html_diagnostic(result, culture_input, image_info, timestamp, location_input)
                        st.download_button(
                            label=t("download_html"),
                            data=html_content,
                            file_name=t("html_filename").format(date=datetime.now().strftime('%Y%m%d_%H%M')),
                            mime="text/html",
                            help=t("export_html")
                        )
                    
                    with col2:
                        # Export Texte
                        text_content = generate_text_diagnostic(result, culture_input, image_info, timestamp, location_input)
                        st.download_button(
                            label=t("download_text"),
                            data=text_content,
                            file_name=t("text_filename").format(date=datetime.now().strftime('%Y%m%d_%H%M')),
                            mime="text/plain",
                            help=t("export_text")
                        )
        except Exception as e:
            st.error(t("image_processing_general_error") + str(e))

# --- ONGLET 2: ANALYSE DE TEXTE ---
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
            st.error(t("model_not_loaded_error"))
        elif not text_input.strip():
            st.error(t("symptoms_required"))
        else:
            # Notification sonore de début d'analyse
            if st.session_state.get('sound_enabled', True):
                play_start_sound()
                st.info(t("analysis_started"))
            
            # Créer des placeholders pour la progression
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Afficher la barre de progression initiale
            progress_placeholder.progress(0)
            status_placeholder.info("🔍 Préparation de l'analyse...")
            
            # Simuler la progression pendant l'analyse
            import time
            
            # Étapes de progression pour l'analyse de texte
            steps = [
                (20, "Analyse du texte..."),
                (50, "Identification du problème..."),
                (80, "Génération des recommandations..."),
                (100, "Analyse terminée !")
            ]
            
            # Effectuer l'analyse avec mise à jour de la progression
            for progress, status in steps[:-1]:  # Toutes sauf la dernière
                progress_placeholder.progress(progress / 100)
                status_placeholder.info(f"🔍 {status}")
                time.sleep(0.3)  # Petite pause pour voir la progression
            
            # Effectuer l'analyse réelle
            result = analyze_text_multilingual(text_input)
            
            # Finaliser la progression
            progress_placeholder.progress(1.0)
            status_placeholder.success("✅ Analyse terminée !")
            
            # Notification sonore de fin d'analyse
            if st.session_state.get('sound_enabled', True):
                play_completion_sound()
                st.success(t("analysis_completed"))
            
            # Effacer les placeholders après un court délai
            time.sleep(1)
            progress_placeholder.empty()
            status_placeholder.empty()
            
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
                html_content = generate_html_diagnostic(result, None, None, timestamp, None)
                st.download_button(
                    label=t("download_html"),
                    data=html_content,
                    file_name=t("html_filename").format(date=datetime.now().strftime('%Y%m%d_%H%M')),
                    mime="text/html",
                    help=t("export_html")
                )
            
            with col2:
                # Export Texte
                text_content = generate_text_diagnostic(result, None, None, timestamp, None)
                st.download_button(
                    label=t("download_text"),
                    data=text_content,
                    file_name=t("text_filename").format(date=datetime.now().strftime('%Y%m%d_%H%M')),
                    mime="text/plain",
                    help=t("export_text")
                )

# --- ONGLET 3: MANUEL ---
with tab3:
    st.header(t("manual_title"))
    
    # Contenu du manuel selon la langue
    manual_content = {
        "fr": """
        ## 🚀 **GUIDE DE DÉMARRAGE RAPIDE**
        
        ### **Étape 1 : Configuration Initiale**
        1. **Choisir la langue** : Dans la sidebar, sélectionnez Français ou English
        2. **Charger le modèle** : Cliquez sur 'Charger le Modèle Gemma 3n E4B IT'
        3. **Attendre le chargement** : Le processus peut prendre 1-2 minutes
        4. **Vérifier le statut** : Le modèle doit afficher "✅ Chargé"

        ### **Étape 2 : Première Analyse**
        1. **Aller dans l'onglet "📸 Analyse d'Image"**
        2. **Télécharger une photo** d'une plante malade
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
        3. **Clarification culture** : Spécifier le type de plante pour un diagnostic plus précis
        4. **Analyse IA** : Le modèle Gemma 3n analyse l'image
        5. **Résultats** : Diagnostic détaillé avec recommandations
        6. **Export** : Possibilité d'exporter en HTML ou texte

        ### **Clarification de la Culture**
        - **Pourquoi** : Aide le modèle à se concentrer sur les maladies spécifiques à la plante
        - **Exemples** : Tomate, Piment, Maïs, Haricot, Aubergine
        - **Précision** : Améliore significativement la qualité du diagnostic

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

        ## 📄 **FONCTIONNALITÉS D'EXPORT**

        ### **Formats Disponibles**
        - **HTML** : Rapport formaté avec mise en page professionnelle
        - **Texte** : Version simple pour archivage ou partage

        ### **Contenu des Exports**
        - **Informations de l'analyse** : Date, heure, modèle utilisé
        - **Culture spécifiée** : Type de plante analysée
        - **Informations image** : Format et taille (si applicable)
        - **Résultats complets** : Diagnostic détaillé
        - **Avertissements** : Disclaimers légaux

        ### **Utilisation des Exports**
        - **Archivage** : Garder une trace des diagnostics
        - **Partage** : Envoyer à des experts ou collègues
        - **Suivi** : Documenter l'évolution des traitements
        - **Formation** : Exemples pour l'apprentissage

        ## ⚙️ **CONFIGURATION ET PARAMÈTRES**

        ### **Paramètres de Langue**
        - **Français** : Interface et réponses en français
        - **English** : Interface and responses in English
        - **Changement** : Via sidebar, effet immédiat

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
        3. **Culture** : Spécifier "Tomate" dans la clarification
        4. **Analyse** : L'IA identifie le mildiou précoce
        5. **Traitement** : Recommandations de fongicides et mesures préventives
        6. **Export** : Sauvegarder le diagnostic pour suivi

        ### **Scénario 2 : Problème de Nutrition**
        1. **Symptômes** : Feuilles jaunies, croissance ralentie
        2. **Description** : Décrire les conditions de culture
        3. **Analyse** : L'IA suggère une carence en azote
        4. **Solution** : Recommandations d'engrais et d'amendements
        5. **Export** : Documenter pour référence future

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
        - **Export** : Possibilité de sauvegarder les diagnostics

        ### **Recommandations d'Usage**
        1. **Formation** : Former les utilisateurs aux bonnes pratiques
        2. **Validation** : Confirmer les diagnostics critiques avec des experts
        3. **Documentation** : Garder des traces des analyses via l'export
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
        3. **Culture clarification** : Specify plant type for more accurate diagnosis
        4. **AI Analysis** : The Gemma 3n model analyzes the image
        5. **Results** : Detailed diagnosis with recommendations
        6. **Export** : Option to export in HTML or text format

        ### **Culture Clarification**
        - **Why** : Helps the model focus on diseases specific to the plant
        - **Examples** : Tomato, Pepper, Corn, Bean, Eggplant
        - **Accuracy** : Significantly improves diagnostic quality

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

        ## 📄 **EXPORT FEATURES**

        ### **Available Formats**
        - **HTML** : Professionally formatted report
        - **Text** : Simple version for archiving or sharing

        ### **Export Content**
        - **Analysis information** : Date, time, model used
        - **Specified culture** : Type of plant analyzed
        - **Image information** : Format and size (if applicable)
        - **Complete results** : Detailed diagnosis
        - **Warnings** : Legal disclaimers

        ### **Using Exports**
        - **Archiving** : Keep track of diagnoses
        - **Sharing** : Send to experts or colleagues
        - **Monitoring** : Document treatment progress
        - **Training** : Examples for learning

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
        3. **Culture** : Specify "Tomato" in clarification
        4. **Analysis** : AI identifies early blight
        5. **Treatment** : Fungicide recommendations and preventive measures
        6. **Export** : Save diagnosis for monitoring

        ### **Scenario 2: Nutrition Problem**
        1. **Symptoms** : Yellowed leaves, slowed growth
        2. **Description** : Describe growing conditions
        3. **Analysis** : AI suggests nitrogen deficiency
        4. **Solution** : Fertilizer and amendment recommendations
        5. **Export** : Document for future reference

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
        - **Export** : Ability to save diagnoses

        ### **Usage Recommendations**
        1. **Training** : Train users in best practices
        2. **Validation** : Confirm critical diagnoses with experts
        3. **Documentation** : Keep records of analyses via export
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

# --- ONGLET 4: À PROPOS ---
with tab4:
    st.header(t("about_title"))
    st.markdown(t("mission_title"))
    st.markdown(t("mission_text"))
    
    st.markdown(t("features_title"))
    st.markdown(t("features_text"))
    
    st.markdown(t("technology_title"))
    
    # Détecter si le modèle local est présent pour adapter le texte
    # is_local = os.path.exists(LOCAL_MODEL_PATH) # Si LOCAL_MODEL_PATH est défini et utilisé
    is_local = False # Pour l'instant, on assume chargement HF
    
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

# --- PIED DE PAGE ---
st.markdown("---")
st.markdown(t("footer"))

# Fermer les divs selon le mode
if is_mobile():
    st.markdown('</div>', unsafe_allow_html=True)  # Fermer mobile-container
else:
    st.markdown('</div>', unsafe_allow_html=True)  # Fermer desktop-container