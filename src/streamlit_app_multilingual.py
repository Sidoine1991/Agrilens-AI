import streamlit as st
import os
import io
from PIL import Image
import requests
import torch
import google.generativeai as genai
import gc
import time
import sys

# Cache global pour le modèle (persiste entre les reruns)
if 'global_model_cache' not in st.session_state:
    st.session_state.global_model_cache = {}
if 'model_load_time' not in st.session_state:
    st.session_state.model_load_time = None
if 'model_persistence_check' not in st.session_state:
    st.session_state.model_persistence_check = False

def check_model_persistence():
    """Vérifie si le modèle est toujours persistant en mémoire"""
    try:
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            # Test simple pour vérifier que le modèle fonctionne
            if hasattr(st.session_state.model, 'device'):
                device = st.session_state.model.device
                return True
        return False
    except Exception:
        return False

def force_model_persistence():
    """Force la persistance du modèle en mémoire"""
    try:
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            # Créer une référence forte au modèle
            st.session_state.global_model_cache['model'] = st.session_state.model
            st.session_state.global_model_cache['processor'] = st.session_state.processor
            st.session_state.global_model_cache['load_time'] = time.time()
            
            # Vérification immédiate
            if st.session_state.global_model_cache.get('model') is not None:
                st.session_state.model_persistence_check = True
                return True
        return False
    except Exception as e:
        st.error(f"Erreur lors de la persistance forcée : {e}")
        return False

def restore_model_from_cache():
    """Restaure le modèle depuis le cache global"""
    try:
        if 'model' in st.session_state.global_model_cache:
            st.session_state.model = st.session_state.global_model_cache['model']
            st.session_state.processor = st.session_state.global_model_cache['processor']
            st.session_state.model_loaded = True
            st.session_state.model_status = "Chargé (cache)"
            return True
        return False
    except Exception:
        return False

def diagnose_loading_issues():
    """Diagnostique les problèmes potentiels de chargement"""
    issues = []
    
    # Vérifier l'environnement
    if os.path.exists("D:/Dev/model_gemma"):
        issues.append("✅ Modèle local détecté")
    else:
        issues.append("🌐 Mode Hugging Face détecté")
    
    # Vérifier les dépendances
    try:
        import transformers
        issues.append(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        issues.append("❌ Transformers non installé")
    
    try:
        import torch
        issues.append(f"✅ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            issues.append(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            issues.append("⚠️ CUDA non disponible - utilisation CPU")
    except ImportError:
        issues.append("❌ PyTorch non installé")
    
    # Vérifier la mémoire disponible
    try:
        import psutil
        memory = psutil.virtual_memory()
        issues.append(f"💾 Mémoire disponible: {memory.available // (1024**3)} GB")
    except ImportError:
        issues.append("⚠️ Impossible de vérifier la mémoire")
    
    return issues

def resize_image_if_needed(image, max_size=(800, 800)):
    """
    Redimensionne l'image si elle dépasse la taille maximale spécifiée
    """
    width, height = image.size
    
    if width > max_size[0] or height > max_size[1]:
        # Calculer le ratio pour maintenir les proportions
        ratio = min(max_size[0] / width, max_size[1] / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Redimensionner l'image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return resized_image, True  # True indique que l'image a été redimensionnée
    else:
        return image, False  # False indique que l'image n'a pas été redimensionnée

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - Plant Disease Diagnosis",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Configuration pour éviter les erreurs 403
st.markdown("""
<script>
// Désactiver les vérifications CORS pour les uploads
window.addEventListener('load', function() {
    if (typeof window.parent !== 'undefined' && window.parent !== window) {
        // Si l'app est dans un iframe (comme sur Hugging Face Spaces)
        console.log('Application détectée dans un iframe - configuration spéciale activée');
    }
});
</script>
""", unsafe_allow_html=True)

# CSS pour mobile
st.markdown("""
<style>
@media (max-width: 600px) {
    .main {
        max-width: 100vw !important;
        padding: 0.5rem !important;
    }
    .stButton button, .stTextInput input, .stTextArea textarea {
        width: 100% !important;
        font-size: 1.1rem !important;
    }
    .stSidebar {
        width: 100vw !important;
        min-width: 100vw !important;
    }
    .result-box {
        font-size: 1.05rem !important;
    }
    .stMarkdown, .stHeader, .stSubheader {
        font-size: 1.1rem !important;
    }
    .stFileUploader, .stImage {
        width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialisation des variables de session
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_status' not in st.session_state:
    st.session_state.model_status = "Non chargé"
if 'language' not in st.session_state:
    st.session_state.language = "fr"

# Configuration Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

# Dictionnaires de traductions
translations = {
    "fr": {
        "title": "🌱 AgriLens AI - Diagnostic des Plantes",
        "subtitle": "**Application de diagnostic des maladies de plantes avec IA**",
        "config_title": "⚙️ Configuration",
        "load_model": "Charger le modèle Gemma 3n E4B IT",
        "model_status": "**Statut du modèle :**",
        "not_loaded": "Non chargé",
        "loaded": "✅ Chargé",
        "error": "❌ Erreur",
        "tabs": ["📸 Analyse d'Image", "💬 Analyse de Texte", "📖 Manuel", "ℹ️ À propos"],
        "image_analysis_title": "🔍 Diagnostic par Image",
        "image_analysis_desc": "Téléchargez une photo de plante malade pour obtenir un diagnostic",
        "choose_image": "Choisissez une image...",
        "analyze_button": "🔬 Analyser avec l'IA",
        "text_analysis_title": "💬 Diagnostic par Texte",
        "text_analysis_desc": "Décrivez les symptômes de votre plante pour obtenir des conseils",
        "symptoms_desc": "Description des symptômes :",
        "analysis_results": "## 📊 Résultats de l'Analyse",
        "manual_title": "📖 Manuel Utilisateur",
        "about_title": "ℹ️ À propos d'AgriLens AI",
        "creator_title": "👨‍💻 Créateur de l'Application",
        "creator_name": "**Sidoine Kolaolé YEBADOKPO**",
        "creator_location": "Bohicon, République du Bénin",
        "creator_phone": "+229 01 96 91 13 46",
        "creator_email": "syebadokpo@gmail.com",
        "creator_linkedin": "linkedin.com/in/sidoineko",
        "creator_portfolio": "Hugging Face Portfolio: Sidoineko/portfolio",
        "competition_title": "🏆 Version Compétition Kaggle",
        "competition_text": "Cette première version d'AgriLens AI a été développée spécifiquement pour participer à la compétition Kaggle. Elle représente notre première production publique et démontre notre expertise en IA appliquée à l'agriculture.",
        "footer": "*AgriLens AI - Diagnostic intelligent des plantes avec IA*"
    },
    "en": {
        "title": "🌱 AgriLens AI - Plant Disease Diagnosis",
        "subtitle": "**AI-powered plant disease diagnosis application**",
        "config_title": "⚙️ Configuration",
        "load_model": "Load Gemma 3n E4B IT Model",
        "model_status": "**Model Status:**",
        "not_loaded": "Not loaded",
        "loaded": "✅ Loaded",
        "error": "❌ Error",
        "tabs": ["📸 Image Analysis", "💬 Text Analysis", "📖 Manual", "ℹ️ About"],
        "image_analysis_title": "🔍 Image Diagnosis",
        "image_analysis_desc": "Upload a photo of a diseased plant to get a diagnosis",
        "choose_image": "Choose an image...",
        "analyze_button": "🔬 Analyze with AI",
        "text_analysis_title": "💬 Text Diagnosis",
        "text_analysis_desc": "Describe your plant symptoms to get advice",
        "symptoms_desc": "Symptom description:",
        "analysis_results": "## 📊 Analysis Results",
        "manual_title": "📖 User Manual",
        "about_title": "ℹ️ About AgriLens AI",
        "creator_title": "👨‍💻 Application Creator",
        "creator_name": "**Sidoine Kolaolé YEBADOKPO**",
        "creator_location": "Bohicon, Benin Republic",
        "creator_phone": "+229 01 96 91 13 46",
        "creator_email": "syebadokpo@gmail.com",
        "creator_linkedin": "linkedin.com/in/sidoineko",
        "creator_portfolio": "Hugging Face Portfolio: Sidoineko/portfolio",
        "competition_title": "🏆 Kaggle Competition Version",
        "competition_text": "This first version of AgriLens AI was specifically developed to participate in the Kaggle competition. It represents our first public production and demonstrates our expertise in AI applied to agriculture.",
        "footer": "*AgriLens AI - Intelligent plant diagnosis with AI*"
    }
}

def t(key):
    return translations[st.session_state.language][key]

def load_model():
    """Charge le modèle Gemma 3n E4B IT selon l'environnement (local ou Hugging Face)"""
    try:
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        
        # Diagnostic initial
        st.info("🔍 Diagnostic de l'environnement...")
        issues = diagnose_loading_issues()
        with st.expander("📊 Diagnostic système", expanded=False):
            for issue in issues:
                st.write(issue)
        
        # Nettoyer la mémoire avant le chargement
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Détecter l'environnement
        is_local = os.path.exists("D:/Dev/model_gemma")
        
        if is_local:
            # Mode LOCAL - Utiliser le modèle téléchargé
            st.info("Chargement du modèle Gemma 3n E4B IT depuis D:/Dev/model_gemma (mode local)...")
            model_path = "D:/Dev/model_gemma"
            
            # Charger le processeur
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Stratégies de chargement pour le mode local
            def load_local_ultra_conservative():
                st.info("Chargement local ultra-conservateur (CPU uniquement, sans device_map)...")
                return Gemma3nForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            def load_local_conservative():
                st.info("Chargement local conservateur (device_map CPU)...")
                return Gemma3nForConditionalGeneration.from_pretrained(
                    model_path,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            # Essayer les stratégies de chargement local
            strategies = [load_local_ultra_conservative, load_local_conservative]
            
            for strategy in strategies:
                try:
                    model = strategy()
                    st.success("Modèle Gemma 3n E4B IT chargé avec succès depuis le dossier local !")
                    return model, processor
                except Exception as e:
                    error_msg = str(e)
                    if "disk_offload" in error_msg.lower():
                        st.warning(f"Stratégie {strategy.__name__} échouée (disk_offload). Tentative suivante...")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    elif "out of memory" in error_msg.lower():
                        st.warning(f"Stratégie {strategy.__name__} échouée (mémoire insuffisante). Tentative suivante...")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        st.warning(f"Stratégie {strategy.__name__} échouée : {error_msg}. Tentative suivante...")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
            
            # Si toutes les stratégies échouent
            st.error("Toutes les stratégies de chargement local ont échoué.")
            return None, None
            
        else:
            # Mode HUGGING FACE - Utiliser le modèle en ligne
            st.info("Chargement du modèle Gemma 3n E4B IT depuis Hugging Face (mode en ligne)...")
            model_id = "google/gemma-3n-E4B-it"
            
            # Charger le processeur
            try:
                st.info("Téléchargement du processeur depuis Hugging Face...")
                processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )
                st.success("Processeur téléchargé avec succès !")
            except Exception as e:
                st.error(f"Erreur lors du téléchargement du processeur : {e}")
                st.info("Tentative de téléchargement avec cache...")
                try:
                    processor = AutoProcessor.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        cache_dir="./cache"
                    )
                    st.success("Processeur téléchargé avec cache !")
                except Exception as e2:
                    st.error(f"Erreur fatale lors du téléchargement du processeur : {e2}")
                    return None, None
            
            # Stratégie 1: Chargement ultra-conservateur (CPU uniquement, sans device_map)
            def load_ultra_conservative():
                st.info("Chargement ultra-conservateur (CPU uniquement, sans device_map)...")
                return Gemma3nForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            # Stratégie 2: Chargement conservateur avec device_map CPU
            def load_conservative():
                st.info("Chargement conservateur (device_map CPU)...")
                return Gemma3nForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            # Stratégie 3: Chargement avec 8-bit quantization
            def load_8bit():
                st.info("Chargement avec quantification 8-bit...")
                return Gemma3nForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    load_in_8bit=True
                )
            
            # Stratégie 4: Chargement avec 4-bit quantization
            def load_4bit():
                st.info("Chargement avec quantification 4-bit...")
                return Gemma3nForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            
            # Stratégie 5: Chargement avec gestion mémoire personnalisée (sans max_memory)
            def load_custom_memory():
                st.info("Chargement avec gestion mémoire personnalisée...")
                return Gemma3nForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            # Vérifier la mémoire disponible
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                st.info(f"Mémoire GPU disponible : {gpu_memory:.1f} GB")
                
                # Essayer différentes stratégies selon la mémoire disponible
                strategies = []
                
                if gpu_memory >= 8:
                    strategies = [load_custom_memory, load_4bit, load_8bit, load_conservative, load_ultra_conservative]
                elif gpu_memory >= 4:
                    strategies = [load_4bit, load_8bit, load_conservative, load_ultra_conservative]
                else:
                    strategies = [load_8bit, load_conservative, load_ultra_conservative]
                
                # Essayer chaque stratégie jusqu'à ce qu'une fonctionne
                for i, strategy in enumerate(strategies):
                    try:
                        st.info(f"Tentative {i+1}/{len(strategies)} : {strategy.__name__}")
                        model = strategy()
                        st.success(f"Modèle chargé avec succès via {strategy.__name__} !")
                        return model, processor
                    except Exception as e:
                        error_msg = str(e)
                        if "disk_offload" in error_msg.lower():
                            st.warning(f"Stratégie {strategy.__name__} échouée (disk_offload). Tentative suivante...")
                            # Nettoyer la mémoire avant la prochaine tentative
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        elif "out of memory" in error_msg.lower():
                            st.warning(f"Stratégie {strategy.__name__} échouée (mémoire insuffisante). Tentative suivante...")
                            # Nettoyer la mémoire avant la prochaine tentative
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            st.warning(f"Stratégie {strategy.__name__} échouée : {error_msg}. Tentative suivante...")
                            # Nettoyer la mémoire avant la prochaine tentative
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                
                # Si toutes les stratégies ont échoué
                st.error("Toutes les stratégies de chargement ont échoué.")
                return None, None
                
            else:
                # Mode CPU uniquement - essayer plusieurs stratégies
                st.warning("GPU non disponible, utilisation du CPU (plus lent)")
                cpu_strategies = [load_ultra_conservative, load_conservative]
                
                for i, strategy in enumerate(cpu_strategies):
                    try:
                        st.info(f"Tentative CPU {i+1}/{len(cpu_strategies)} : {strategy.__name__}")
                        model = strategy()
                        st.success(f"Modèle chargé avec succès en mode CPU via {strategy.__name__} !")
                        return model, processor
                    except Exception as e:
                        error_msg = str(e)
                        if "disk_offload" in error_msg.lower():
                            st.warning(f"Stratégie CPU {strategy.__name__} échouée (disk_offload). Tentative suivante...")
                            gc.collect()
                            continue
                        else:
                            st.warning(f"Stratégie CPU {strategy.__name__} échouée : {error_msg}. Tentative suivante...")
                            gc.collect()
                            continue
                
                st.error("Toutes les stratégies de chargement CPU ont échoué.")
                return None, None
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None

def analyze_image_multilingual(image, prompt=""):
    """Analyse une image avec Gemma 3n E4B IT pour diagnostic précis"""
    try:
        # Vérification complète du modèle avec cache
        if not st.session_state.model_loaded and not check_model_persistence():
            # Essayer de restaurer depuis le cache
            if restore_model_from_cache():
                st.info("🔄 Modèle restauré depuis le cache pour l'analyse")
            else:
                return "❌ Modèle Gemma non chargé. Veuillez d'abord charger le modèle dans les réglages."
        
        # Vérifier que le modèle et le processeur sont disponibles
        if not hasattr(st.session_state, 'model') or st.session_state.model is None:
            # Essayer de restaurer depuis le cache
            if restore_model_from_cache():
                st.info("🔄 Modèle restauré depuis le cache")
            else:
                st.session_state.model_loaded = False
                return "❌ Modèle perdu en mémoire. Veuillez recharger le modèle."
        
        if not hasattr(st.session_state, 'processor') or st.session_state.processor is None:
            st.session_state.model_loaded = False
            return "❌ Processeur perdu en mémoire. Veuillez recharger le modèle."
        
        # Récupérer le modèle et le processeur
        model, processor = st.session_state.model, st.session_state.processor
        
        # Vérification finale
        if not model or not processor:
            st.session_state.model_loaded = False
            return "❌ Modèle Gemma non disponible. Veuillez recharger le modèle."
        
        # Préparer le prompt pour Gemma 3n
        if st.session_state.language == "fr":
            if prompt:
                gemma_prompt = f"""
Tu es un expert en pathologie végétale. Analyse cette image de plante et fournis un diagnostic précis.

**Question spécifique :** {prompt}

**Instructions :**
1. **Diagnostic précis** : Identifie la maladie spécifique avec son nom scientifique
2. **Causes** : Explique les causes probables (champignons, bactéries, virus, carences, etc.)
3. **Symptômes détaillés** : Liste tous les symptômes observables dans l'image
4. **Traitement spécifique** : Donne des recommandations de traitement précises
5. **Actions préventives** : Conseils pour éviter la propagation
6. **Urgence** : Indique si c'est urgent ou non

**Format de réponse :**
## 🔍 **Diagnostic Précis**
[Nom de la maladie et causes]

## 📋 **Symptômes Détaillés**
[Liste des symptômes observés]

## 💊 **Traitement Recommandé**
[Actions spécifiques à entreprendre]

## 🛡️ **Actions Préventives**
[Mesures pour éviter la propagation]

## ⚠️ **Niveau d'Urgence**
[Urgent/Modéré/Faible]

Réponds de manière structurée et précise.
"""
            else:
                gemma_prompt = """
Tu es un expert en pathologie végétale. Analyse cette image de plante et fournis un diagnostic précis.

**Instructions :**
1. **Diagnostic précis** : Identifie la maladie spécifique avec son nom scientifique
2. **Causes** : Explique les causes probables (champignons, bactéries, virus, carences, etc.)
3. **Symptômes détaillés** : Liste tous les symptômes observables dans l'image
4. **Traitement spécifique** : Donne des recommandations de traitement précises
5. **Actions préventives** : Conseils pour éviter la propagation
6. **Urgence** : Indique si c'est urgent ou non

**Format de réponse :**
## 🔍 **Diagnostic Précis**
[Nom de la maladie et causes]

## 📋 **Symptômes Détaillés**
[Liste des symptômes observés]

## 💊 **Traitement Recommandé**
[Actions spécifiques à entreprendre]

## 🛡️ **Actions Préventives**
[Mesures pour éviter la propagation]

## ⚠️ **Niveau d'Urgence**
[Urgent/Modéré/Faible]

Réponds de manière structurée et précise.
"""
        else:
            if prompt:
                gemma_prompt = f"""
You are an expert in plant pathology. Analyze this plant image and provide a precise diagnosis.

**Specific Question:** {prompt}

**Instructions:**
1. **Precise Diagnosis**: Identify the specific disease with its scientific name
2. **Causes**: Explain probable causes (fungi, bacteria, viruses, deficiencies, etc.)
3. **Detailed Symptoms**: List all observable symptoms in the image
4. **Specific Treatment**: Give precise treatment recommendations
5. **Preventive Actions**: Advice to prevent spread
6. **Urgency**: Indicate if urgent or not

**Response Format:**
## 🔍 **Precise Diagnosis**
[Disease name and causes]

## 📋 **Detailed Symptoms**
[List of observed symptoms]

## 💊 **Recommended Treatment**
[Specific actions to take]

## 🛡️ **Preventive Actions**
[Measures to prevent spread]

## ⚠️ **Urgency Level**
[Urgent/Moderate/Low]

Respond in a structured and precise manner.
"""
            else:
                gemma_prompt = """
You are an expert in plant pathology. Analyze this plant image and provide a precise diagnosis.

**Instructions:**
1. **Precise Diagnosis**: Identify the specific disease with its scientific name
2. **Causes**: Explain probable causes (fungi, bacteria, viruses, deficiencies, etc.)
3. **Detailed Symptoms**: List all observable symptoms in the image
4. **Specific Treatment**: Give precise treatment recommendations
5. **Preventive Actions**: Advice to prevent spread
6. **Urgency**: Indicate if urgent or not

**Response Format:**
## 🔍 **Precise Diagnosis**
[Disease name and causes]

## 📋 **Detailed Symptoms**
[List of observed symptoms]

## 💊 **Recommended Treatment**
[Specific actions to take]

## 🛡️ **Preventive Actions**
[Measures to prevent spread]

## ⚠️ **Urgency Level**
[Urgent/Moderate/Low]

Respond in a structured and precise manner.
"""
        
        # Préparer les messages pour Gemma 3n
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert in plant pathology."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": gemma_prompt}
                ]
            }
        ]
        
        # Traiter les entrées avec le processeur
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
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
            generation = generation[0][input_len:]
        
        # Décoder la réponse
        response_text = processor.decode(generation, skip_special_tokens=True)
        
        if st.session_state.language == "fr":
            return f"""
## 🧠 **Analyse par Gemma 3n E4B IT**
{response_text}
"""
        else:
            return f"""
## 🧠 **Analysis by Gemma 3n E4B IT**
{response_text}
"""
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse d'image : {e}"

def analyze_text_multilingual(text):
    """Analyse un texte avec le modèle Gemma 3n E4B IT"""
    # Vérification complète du modèle avec cache
    if not st.session_state.model_loaded and not check_model_persistence():
        # Essayer de restaurer depuis le cache
        if restore_model_from_cache():
            st.info("🔄 Modèle restauré depuis le cache pour l'analyse")
        else:
            return "❌ Modèle non chargé. Veuillez le charger dans les réglages."
    
    # Vérifier que le modèle et le processeur sont disponibles
    if not hasattr(st.session_state, 'model') or st.session_state.model is None:
        # Essayer de restaurer depuis le cache
        if restore_model_from_cache():
            st.info("🔄 Modèle restauré depuis le cache")
        else:
            st.session_state.model_loaded = False
            return "❌ Modèle perdu en mémoire. Veuillez recharger le modèle."
    
    if not hasattr(st.session_state, 'processor') or st.session_state.processor is None:
        st.session_state.model_loaded = False
        return "❌ Processeur perdu en mémoire. Veuillez recharger le modèle."
    
    try:
        model, processor = st.session_state.model, st.session_state.processor
        
        # Vérification finale
        if not model or not processor:
            st.session_state.model_loaded = False
            return "❌ Modèle Gemma non disponible. Veuillez recharger le modèle."
        
        if st.session_state.language == "fr":
            prompt = f"Tu es un assistant agricole expert. Analyse ce problème : {text}"
        else:
            prompt = f"You are an expert agricultural assistant. Analyze this problem: {text}"
        
        # Préparer les messages
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        
        # Traiter les entrées
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Générer la réponse
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            generation = generation[0][input_len:]
        
        response = processor.decode(generation, skip_special_tokens=True)
        return response
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse de texte : {e}"

def analyze_with_gemini(gemma_description, image_info=""):
    """
    Analyse approfondie avec Gemini API pour diagnostic précis
    """
    if not gemini_model:
        return "❌ Gemini API non configurée. Vérifiez votre clé API Google."
    
    try:
        if st.session_state.language == "fr":
            prompt = f"""
Tu es un expert en pathologie végétale et en agriculture. Analyse cette description d'une plante malade et fournis un diagnostic précis.

**Description de Gemma :**
{gemma_description}

**Informations supplémentaires :**
{image_info}

**Instructions :**
1. **Diagnostic précis** : Identifie la maladie spécifique avec son nom scientifique
2. **Causes** : Explique les causes probables (champignons, bactéries, virus, carences, etc.)
3. **Symptômes détaillés** : Liste tous les symptômes observables
4. **Traitement spécifique** : Donne des recommandations de traitement précises
5. **Actions préventives** : Conseils pour éviter la propagation
6. **Urgence** : Indique si c'est urgent ou non

**Format de réponse :**
## 🔍 **Diagnostic Précis**
[Nom de la maladie et causes]

## 📋 **Symptômes Détaillés**
[Liste des symptômes]

## 💊 **Traitement Recommandé**
[Actions spécifiques à entreprendre]

## 🛡️ **Actions Préventives**
[Mesures pour éviter la propagation]

## ⚠️ **Niveau d'Urgence**
[Urgent/Modéré/Faible]

Réponds de manière structurée et précise.
"""
        else:
            prompt = f"""
You are an expert in plant pathology and agriculture. Analyze this description of a diseased plant and provide a precise diagnosis.

**Gemma's Description:**
{gemma_description}

**Additional Information:**
{image_info}

**Instructions:**
1. **Precise Diagnosis**: Identify the specific disease with its scientific name
2. **Causes**: Explain probable causes (fungi, bacteria, viruses, deficiencies, etc.)
3. **Detailed Symptoms**: List all observable symptoms
4. **Specific Treatment**: Give precise treatment recommendations
5. **Preventive Actions**: Advice to prevent spread
6. **Urgency**: Indicate if urgent or not

**Response Format:**
## 🔍 **Precise Diagnosis**
[Disease name and causes]

## 📋 **Detailed Symptoms**
[List of symptoms]

## 💊 **Recommended Treatment**
[Specific actions to take]

## 🛡️ **Preventive Actions**
[Measures to prevent spread]

## ⚠️ **Urgency Level**
[Urgent/Moderate/Low]

Respond in a structured and precise manner.
"""
        
        response = gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse Gemini : {e}"

# Interface principale
st.title(t("title"))
st.markdown(t("subtitle"))

# Sidebar pour la configuration
with st.sidebar:
    st.header(t("config_title"))
    
    # Sélecteur de langue
    st.subheader("🌐 Sélection de langue / Language Selection")
    language = st.selectbox(
        "Language / Langue",
        ["Français", "English"],
        index=0 if st.session_state.language == "fr" else 1
    )
    
    if language == "Français":
        st.session_state.language = "fr"
    else:
        st.session_state.language = "en"
    
    # Vérification automatique de la persistance du modèle
    if not st.session_state.model_loaded and not check_model_persistence():
        # Essayer de restaurer depuis le cache
        if restore_model_from_cache():
            st.success("✅ Modèle restauré depuis le cache")
        else:
            st.info("🔄 Modèle non trouvé en cache - chargement nécessaire")
    
    # Chargement du modèle
    if st.button(t("load_model"), type="primary"):
        with st.spinner("Chargement du modèle..." if st.session_state.language == "fr" else "Loading model..."):
            try:
                model, processor = load_model()
                if model and processor:
                    # Stocker le modèle dans la session avec vérification
                    st.session_state.model = model
                    st.session_state.processor = processor
                    st.session_state.model_loaded = True
                    st.session_state.model_status = t("loaded")
                    st.session_state.model_load_time = time.time()
                    
                    # Forcer la persistance en cache global
                    if force_model_persistence():
                        if is_local:
                            st.success("✅ Modèle Gemma 3n E4B IT chargé et persisté avec succès (mode local) !")
                            st.info("🔄 Le modèle local est maintenant sauvegardé en cache pour la persistance")
                        else:
                            st.success("✅ Modèle Gemma 3n E4B IT chargé et persisté avec succès (mode Hugging Face) !")
                            st.info("🔄 Le modèle en ligne est maintenant sauvegardé en cache pour la persistance")
                    else:
                        st.warning("⚠️ Modèle chargé mais problème de persistance détecté")
                        
                else:
                    st.session_state.model_loaded = False
                    st.session_state.model_status = t("error")
                    st.error("Échec du chargement du modèle" if st.session_state.language == "fr" else "Model loading failed")
            except Exception as e:
                st.session_state.model_loaded = False
                st.session_state.model_status = t("error")
                st.error(f"Erreur lors du chargement : {e}")
    
    st.info(f"{t('model_status')} {st.session_state.model_status}")
    
    # Vérification de la persistance du modèle
    if st.session_state.model_loaded or check_model_persistence():
        # Vérifier que le modèle est toujours disponible
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            if hasattr(st.session_state, 'processor') and st.session_state.processor is not None:
                # Détecter l'environnement pour l'affichage
                is_local = os.path.exists("D:/Dev/model_gemma")
                
                # Afficher le statut de persistance
                if st.session_state.model_persistence_check:
                    if is_local:
                        st.success("✅ Modèle Gemma 3n E4B IT chargé et PERSISTANT (mode local - cache activé)")
                        st.info("🔄 Le modèle local est sauvegardé en cache et persiste entre les sessions")
                    else:
                        st.success("✅ Modèle Gemma 3n E4B IT chargé et PERSISTANT (mode Hugging Face - cache activé)")
                        st.info("🔄 Le modèle en ligne est sauvegardé en cache et persiste entre les sessions")
                else:
                    if is_local:
                        st.success("✅ Modèle Gemma 3n E4B IT chargé (mode local)")
                        st.info("Le modèle local est prêt pour l'analyse d'images et de texte")
                    else:
                        st.success("✅ Modèle Gemma 3n E4B IT chargé (mode Hugging Face)")
                        st.info("Le modèle en ligne est prêt pour l'analyse d'images et de texte")
                
                # Diagnostic du modèle
                with st.expander("🔍 Diagnostic du modèle"):
                    st.write(f"**Mode d'exécution :** {'🏠 Local (D:/Dev/model_gemma)' if is_local else '🌐 Hugging Face (en ligne)'}")
                    st.write(f"**Modèle chargé :** {type(st.session_state.model).__name__}")
                    st.write(f"**Processeur chargé :** {type(st.session_state.processor).__name__}")
                    st.write(f"**Device du modèle :** {st.session_state.model.device}")
                    st.write(f"**Mémoire utilisée :** {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "CPU uniquement")
                    if st.session_state.model_load_time:
                        load_time_str = time.strftime('%H:%M:%S', time.localtime(st.session_state.model_load_time))
                        st.write(f"**Heure de chargement :** {load_time_str}")
                    st.write(f"**Cache global :** {'✅ Actif' if st.session_state.global_model_cache else '❌ Inactif'}")
                
                # Boutons de gestion
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Recharger le modèle", type="secondary"):
                        st.session_state.model_loaded = False
                        st.session_state.model = None
                        st.session_state.processor = None
                        st.session_state.global_model_cache.clear()
                        st.rerun()
                with col2:
                    if st.button("💾 Forcer la persistance", type="secondary"):
                        if force_model_persistence():
                            st.success("✅ Persistance forcée avec succès")
                        else:
                            st.error("❌ Échec de la persistance forcée")
                        st.rerun()
            else:
                st.warning("⚠️ Processeur manquant - rechargement nécessaire")
                st.session_state.model_loaded = False
                st.session_state.model = None
        else:
            st.warning("⚠️ Modèle perdu en mémoire - rechargement nécessaire")
            st.session_state.model_loaded = False
            st.session_state.processor = None
    else:
        # Détecter l'environnement pour l'affichage
        is_local = os.path.exists("D:/Dev/model_gemma")
        
        st.warning("⚠️ Modèle Gemma 3n E4B IT non chargé")
        if is_local:
            st.info("Cliquez sur 'Charger le modèle' pour activer l'analyse (mode local)")
        else:
            st.info("Cliquez sur 'Charger le modèle' pour activer l'analyse (mode Hugging Face)")

# Onglets principaux
tab1, tab2, tab3, tab4 = st.tabs(t("tabs"))

with tab1:
    st.header(t("image_analysis_title"))
    st.markdown(t("image_analysis_desc"))
    
    # Options de capture d'image
    capture_option = st.radio(
        "Choisissez votre méthode :" if st.session_state.language == "fr" else "Choose your method:",
        ["📁 Upload d'image" if st.session_state.language == "fr" else "📁 Upload Image", 
         "📷 Capture par webcam" if st.session_state.language == "fr" else "📷 Webcam Capture"],
        horizontal=True
    )
    
    uploaded_file = None
    captured_image = None
    
    if capture_option == "📁 Upload d'image" or capture_option == "📁 Upload Image":
        try:
            uploaded_file = st.file_uploader(
                t("choose_image"), 
                type=['png', 'jpg', 'jpeg'],
                help="Formats acceptés : PNG, JPG, JPEG (max 200MB)" if st.session_state.language == "fr" else "Accepted formats: PNG, JPG, JPEG (max 200MB)",
                accept_multiple_files=False,
                key="image_uploader"
            )
        except Exception as e:
            st.error(f"❌ Erreur lors de l'upload : {e}")
    
    else:  # Webcam capture
        st.markdown("**📷 Capture d'image par webcam**" if st.session_state.language == "fr" else "**📷 Webcam Image Capture**")
        st.info("💡 Positionnez votre plante malade devant la webcam et cliquez sur 'Prendre une photo'" if st.session_state.language == "fr" else "💡 Position your diseased plant in front of the webcam and click 'Take Photo'")
        
        captured_image = st.camera_input(
            "Prendre une photo de la plante" if st.session_state.language == "fr" else "Take a photo of the plant",
            key="webcam_capture"
        )
    
    # Traitement de l'image (upload ou webcam)
    image = None
    image_source = None
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_source = "upload"
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement de l'image uploadée : {e}")
            st.info("💡 Essayez avec une image différente ou un format différent (PNG, JPG, JPEG)")
    elif captured_image is not None:
        try:
            image = Image.open(captured_image)
            image_source = "webcam"
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement de l'image capturée : {e}")
            st.info("💡 Essayez de reprendre la photo")
    
    if image is not None:
        try:
            # Redimensionner l'image si nécessaire
            original_size = image.size
            image, was_resized = resize_image_if_needed(image, max_size=(800, 800))
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if image_source == "upload":
                    st.image(image, caption="Image uploadée" if st.session_state.language == "fr" else "Uploaded Image", use_container_width=True)
                else:
                    st.image(image, caption="Image capturée par webcam" if st.session_state.language == "fr" else "Webcam Captured Image", use_container_width=True)
            
            with col2:
                st.markdown("**Informations de l'image :**")
                st.write(f"• Format : {image.format}")
                st.write(f"• Taille originale : {original_size[0]}x{original_size[1]} pixels")
                st.write(f"• Taille actuelle : {image.size[0]}x{image.size[1]} pixels")
                st.write(f"• Mode : {image.mode}")
                
                if was_resized:
                    st.warning("⚠️ L'image a été automatiquement redimensionnée pour optimiser le traitement")
            
            question = st.text_area(
                "Question spécifique (optionnel) :",
                placeholder="Ex: Quelle est cette maladie ? Que faire pour la traiter ?",
                height=100
            )
            
            if st.button(t("analyze_button"), disabled=not st.session_state.model_loaded, type="primary"):
                if not st.session_state.model_loaded:
                    st.error("❌ Modèle Gemma non chargé. Veuillez d'abord charger le modèle dans les réglages.")
                    st.info("💡 L'analyse d'image nécessite le modèle Gemma 3n E4B IT. Chargez-le dans les réglages.")
                else:
                    with st.spinner("🔍 Analyse en cours..."):
                        result = analyze_image_multilingual(image, question)
                    
                    st.markdown(t("analysis_results"))
                    st.markdown("---")
                    st.markdown(result)
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "Forbidden" in error_msg:
                st.error("❌ Erreur 403 - Accès refusé lors du traitement de l'image")
                st.warning("🔒 Cette erreur indique un problème d'autorisation côté serveur.")
                st.info("💡 Solutions possibles :")
                st.info("• Vérifiez les logs de votre espace Hugging Face")
                st.info("• Essayez avec une image plus petite (< 1MB)")
                st.info("• Rafraîchissez la page et réessayez")
                st.info("• Contactez le support Hugging Face si le problème persiste")
            else:
                st.error(f"❌ Erreur lors du traitement de l'image : {e}")
                st.info("💡 Essayez avec une image différente ou un format différent (PNG, JPG, JPEG)")

with tab2:
    st.header(t("text_analysis_title"))
    st.markdown(t("text_analysis_desc"))
    
    text_input = st.text_area(
        t("symptoms_desc"),
        placeholder="Ex: Mes tomates ont des taches brunes sur les feuilles et les fruits...",
        height=150
    )
    
    if st.button("🧠 Analyser avec l'IA", disabled=not st.session_state.model_loaded, type="primary"):
        if not st.session_state.model_loaded:
            st.error("❌ Veuillez d'abord charger le modèle dans les réglages")
        elif not text_input.strip():
            st.error("❌ Veuillez saisir une description")
        else:
            with st.spinner("🔍 Analyse en cours..."):
                result = analyze_text_multilingual(text_input)
            
            st.markdown(t("analysis_results"))
            st.markdown("---")
            st.markdown(result)

with tab3:
    st.header(t("manual_title"))
    
    if st.session_state.language == "fr":
        st.markdown("""
        ### 🚀 **Démarrage Rapide**
        1. **Charger le modèle** : Cliquez sur 'Charger le modèle' dans les réglages
        2. **Choisir le mode** : Analyse d'image ou analyse de texte
        3. **Soumettre votre demande** : Upload d'image ou description
        4. **Obtenir le diagnostic** : Résultats avec recommandations
        
        ### 📸 **Analyse d'Image**
        • **Formats acceptés** : PNG, JPG, JPEG
        • **Taille recommandée** : 500x500 pixels minimum
        • **Qualité** : Image claire et bien éclairée
        • **Focus** : Centrer sur la zone malade
        • **Question optionnelle** : Précisez votre préoccupation
        
        ### 💬 **Analyse de Texte**
        • **Description détaillée** : Symptômes observés
        • **Contexte** : Type de plante, conditions
        • **Historique** : Évolution du problème
        • **Actions déjà tentées** : Traitements appliqués
        
        ### 🔍 **Interprétation des Résultats**
        • **Diagnostic** : Identification de la maladie
        • **Causes possibles** : Facteurs déclencheurs
        • **Recommandations** : Actions à entreprendre
        • **Prévention** : Mesures préventives
        
        ### 💡 **Bonnes Pratiques**
        • **Images multiples** : Différents angles de la maladie
        • **Éclairage naturel** : Éviter les ombres
        • **Description précise** : Détails des symptômes
        • **Suivi régulier** : Surveiller l'évolution
        • **Consultation expert** : Pour cas complexes
        """)
    else:
        st.markdown("""
        ### 🚀 **Quick Start**
        1. **Load the model** : Click 'Load Model' in settings
        2. **Choose mode** : Image analysis or text analysis
        3. **Submit your request** : Upload image or description
        4. **Get diagnosis** : Results with recommendations
        
        ### 📸 **Image Analysis**
        • **Accepted formats** : PNG, JPG, JPEG
        • **Recommended size** : 500x500 pixels minimum
        • **Quality** : Clear and well-lit image
        • **Focus** : Center on the diseased area
        • **Optional question** : Specify your concern
        
        ### 💬 **Text Analysis**
        • **Detailed description** : Observed symptoms
        • **Context** : Plant type, conditions
        • **History** : Problem evolution
        • **Actions already tried** : Applied treatments
        
        ### 🔍 **Result Interpretation**
        • **Diagnosis** : Disease identification
        • **Possible causes** : Triggering factors
        • **Recommendations** : Actions to take
        • **Prevention** : Preventive measures
        
        ### 💡 **Best Practices**
        • **Multiple images** : Different angles of the disease
        • **Natural lighting** : Avoid shadows
        • **Precise description** : Symptom details
        • **Regular monitoring** : Track evolution
        • **Expert consultation** : For complex cases
        """)

with tab4:
    st.header(t("about_title"))
    
    st.markdown("### 🌱 Notre Mission / Our Mission")
    st.markdown("AgriLens AI est une application de diagnostic des maladies de plantes utilisant l'intelligence artificielle pour aider les agriculteurs à identifier et traiter les problèmes de leurs cultures.")
    
    st.markdown("### 🚀 Fonctionnalités / Features")
    st.markdown("""
    • **Analyse d'images** : Diagnostic visuel des maladies
    • **Analyse de texte** : Conseils basés sur les descriptions
    • **Recommandations pratiques** : Actions concrètes à entreprendre
    • **Interface mobile** : Optimisée pour smartphones et tablettes
    • **Support multilingue** : Français et Anglais
    """)
    
    st.markdown("### 🔧 Technologie / Technology")
    
    # Détecter l'environnement pour l'affichage
    is_local = os.path.exists("D:/Dev/model_gemma")
    
    if is_local:
        st.markdown("""
        • **Modèle** : Gemma 3n E4B IT (Local - D:/Dev/model_gemma)
        • **Framework** : Streamlit
        • **Déploiement** : Local
        """)
    else:
        st.markdown("""
        • **Modèle** : Gemma 3n E4B IT (Hugging Face - en ligne)
        • **Framework** : Streamlit
        • **Déploiement** : Hugging Face Spaces
        """)
    
    # Informations du créateur
    st.markdown(f"### {t('creator_title')}")
    st.markdown(f"{t('creator_name')}")
    st.markdown(f"📍 {t('creator_location')}")
    st.markdown(f"📞 {t('creator_phone')}")
    st.markdown(f"📧 {t('creator_email')}")
    st.markdown(f"🔗 {t('creator_linkedin')}")
    st.markdown(f"📁 {t('creator_portfolio')}")
    
    # Informations de compétition
    st.markdown(f"### {t('competition_title')}")
    st.markdown(t("competition_text"))
    
    st.markdown("### ⚠️ Avertissement / Warning")
    st.markdown("Les résultats fournis sont à titre indicatif uniquement. Pour un diagnostic professionnel, consultez un expert qualifié.")
    
    st.markdown("### 📞 Support")
    st.markdown("Pour toute question ou problème, consultez la documentation ou contactez l'équipe de développement.")

# Footer
st.markdown("---")
st.markdown(t("footer")) 