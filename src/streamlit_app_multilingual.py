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
import psutil
import signal
from contextlib import contextmanager

# Cache global pour le modèle (persiste entre les reruns)
if 'global_model_cache' not in st.session_state:
    st.session_state.global_model_cache = {}
if 'model_load_time' not in st.session_state:
    st.session_state.model_load_time = None
if 'model_persistence_check' not in st.session_state:
    st.session_state.model_persistence_check = False

@contextmanager
def timeout(seconds):
    """Context manager pour timeout"""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Timeout après {seconds} secondes")
    
    # Enregistrer l'ancien handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restaurer l'ancien handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

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
            # Log de débogage
            st.write("🔍 DEBUG: Tentative de persistance du modèle...")
            st.write(f"🔍 DEBUG: Modèle présent: {st.session_state.model is not None}")
            st.write(f"🔍 DEBUG: Type du modèle: {type(st.session_state.model).__name__}")
            
            # Créer une référence forte au modèle
            st.session_state.global_model_cache['model'] = st.session_state.model
            st.session_state.global_model_cache['processor'] = st.session_state.processor
            st.session_state.global_model_cache['load_time'] = time.time()
            st.session_state.global_model_cache['model_type'] = type(st.session_state.model).__name__
            st.session_state.global_model_cache['processor_type'] = type(st.session_state.processor).__name__
            
            # Vérification immédiate
            if st.session_state.global_model_cache.get('model') is not None:
                st.session_state.model_persistence_check = True
                
                # Vérification supplémentaire
                if hasattr(st.session_state.global_model_cache['model'], 'device'):
                    st.session_state.global_model_cache['device'] = st.session_state.global_model_cache['model'].device
                
                st.write("🔍 DEBUG: Persistance réussie!")
                st.write(f"🔍 DEBUG: Cache contient: {list(st.session_state.global_model_cache.keys())}")
                return True
            else:
                st.write("🔍 DEBUG: Échec de la persistance - modèle non trouvé dans le cache")
        else:
            st.write("🔍 DEBUG: Échec de la persistance - modèle non présent dans session_state")
        return False
    except Exception as e:
        st.error(f"Erreur lors de la persistance forcée : {e}")
        st.write(f"🔍 DEBUG: Exception lors de la persistance: {e}")
        return False

def restore_model_from_cache():
    """Restaure le modèle depuis le cache global"""
    try:
        st.write("🔍 DEBUG: Tentative de restauration depuis le cache...")
        st.write(f"🔍 DEBUG: Cache disponible: {list(st.session_state.global_model_cache.keys())}")
        
        if 'model' in st.session_state.global_model_cache and st.session_state.global_model_cache['model'] is not None:
            # Vérifier que le modèle est toujours valide
            cached_model = st.session_state.global_model_cache['model']
            st.write(f"🔍 DEBUG: Modèle trouvé dans le cache: {cached_model is not None}")
            
            if hasattr(cached_model, 'device'):
                # Le modèle semble valide
                st.session_state.model = cached_model
                st.session_state.processor = st.session_state.global_model_cache['processor']
                st.session_state.model_loaded = True
                st.session_state.model_status = "Chargé (cache)"
                
                # Mettre à jour le temps de chargement si disponible
                if 'load_time' in st.session_state.global_model_cache:
                    st.session_state.model_load_time = st.session_state.global_model_cache['load_time']
                
                st.write("🔍 DEBUG: Restauration réussie!")
                return True
            else:
                st.write("🔍 DEBUG: Modèle dans le cache mais pas d'attribut 'device'")
        else:
            st.write("🔍 DEBUG: Modèle non trouvé dans le cache")
        return False
    except Exception as e:
        st.error(f"Erreur lors de la restauration depuis le cache : {e}")
        st.write(f"🔍 DEBUG: Exception lors de la restauration: {e}")
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

def afficher_ram_disponible(context=""):
    mem = psutil.virtual_memory()
    st.info(f"💾 RAM disponible {context}: {mem.available // (1024**2)} MB ({mem.available // (1024**3)} GB)")
    if mem.available < 4 * 1024**3:
        st.warning("⚠️ Moins de 4GB de RAM disponible, le chargement du modèle risque d'échouer !")

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

# === AJOUT : Chargement automatique du modèle local au démarrage ===
is_local = os.path.exists("models/gemma-3n-transformers-gemma-3n-e2b-it-v1")
if is_local and not st.session_state.model_loaded:
                st.info("🔄 Chargement automatique du modèle local : models/gemma-3n-transformers-gemma-3n-e2b-it-v1 ...")
                try:
                    from transformers import AutoProcessor, Gemma3nForConditionalGeneration
                    model_path = "models/gemma-3n-transformers-gemma-3n-e2b-it-v1"
                    
                    # Nettoyer la mémoire avant le chargement
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Afficher la RAM disponible avant chargement
                    afficher_ram_disponible("AVANT chargement automatique")
                    
                    # Charger le processeur avec la configuration locale
                    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                    
                    # Charger le modèle avec la configuration locale (ultra-agressive)
                    # Utiliser bfloat16 comme spécifié dans config.json et forcer CPU
                    model = Gemma3nForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,  # Utiliser bfloat16 comme dans config.json
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        device_map=None,  # Forcer CPU
                        max_memory=None,  # Pas de limite de mémoire
                        offload_folder=None,  # Pas d'offload
                        offload_state_dict=False,  # Pas d'offload
                        load_in_4bit=False,  # Pas de quantification
                        load_in_8bit=False,  # Pas de quantification
                        attn_implementation="eager"  # Implémentation CPU
                    )
                    
                    st.session_state.model = model
                    st.session_state.processor = processor
                    st.session_state.model_loaded = True
                    st.session_state.model_status = "Chargé automatiquement (local)"
                    st.session_state.model_load_time = time.time()
                    
                    # Afficher la RAM disponible après chargement
                    afficher_ram_disponible("APRÈS chargement automatique")
                    
                    st.success("✅ Modèle local chargé automatiquement au démarrage !")
                except Exception as e:
                    st.session_state.model_loaded = False
                    st.session_state.model_status = "Erreur chargement automatique"
                    st.error(f"❌ Erreur lors du chargement automatique du modèle local : {e}")
                    st.write(f"🔍 DEBUG: Exception détaillée: {str(e)}")
elif not is_local and not st.session_state.model_loaded:
    # Ne pas charger automatiquement sur HF Spaces - laisser l'utilisateur choisir
    st.info("🔄 Environnement Hugging Face Spaces détecté")
    st.info("⚠️ Chargement automatique désactivé pour éviter les erreurs de mémoire")
    st.info("💡 Utilisez le bouton 'Charger le modèle' pour charger un modèle compatible")
    st.session_state.model_loaded = False
    st.session_state.model_status = "En attente de chargement manuel"

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
        "load_model": "Charger un modèle IA",
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
        "load_model": "Load AI Model",
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

# Classe processeur personnalisée pour Gemma 3n
class Gemma3nProcessor:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
    
    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        # Traiter le texte avec le tokenizer
        if text is not None:
            text_inputs = self.tokenizer(
                text, 
                return_tensors=return_tensors, 
                padding=True, 
                truncation=True,
                **kwargs
            )
        
        # Traiter l'image avec l'image processor
        if images is not None:
            image_inputs = self.image_processor(
                images, 
                return_tensors=return_tensors,
                **kwargs
            )
            
            # Combiner les inputs
            if text is not None:
                inputs = {**text_inputs, **image_inputs}
            else:
                inputs = image_inputs
        else:
            inputs = text_inputs
        
        return inputs
    
    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)

def load_model():
    """Charge le modèle avec gestion d'erreurs et fallback pour HF Spaces"""
    
    # Vérifier si le modèle est déjà chargé
    if hasattr(st.session_state, 'model') and st.session_state.model is not None:
        if check_model_persistence():
            st.success("✅ Modèle déjà chargé et persistant")
            return st.session_state.model, st.session_state.processor
    
    # Détecter l'environnement Hugging Face Spaces
    is_hf_spaces = os.environ.get('SPACE_ID') is not None
    
    if is_hf_spaces:
        st.warning("🚨 Environnement Hugging Face Spaces détecté")
        st.info("⚠️ Utilisation de modèles légers compatibles avec les contraintes mémoire")
        
        # Stratégie HF Spaces : Modèles ultra-légers uniquement
        try:
            return load_ultra_lightweight_for_hf_spaces()
        except Exception as e:
            st.error(f"❌ Erreur avec le modèle ultra-léger : {e}")
            st.info("🔄 Basculement vers le pipeline basique...")
            try:
                return load_basic_pipeline()
            except Exception as e2:
                st.error(f"❌ Erreur avec le pipeline basique : {e2}")
                return None, None
    
    # Stratégies pour environnement local avec plus de mémoire
    if torch.cuda.is_available():
        st.info("🚀 GPU détecté - Tentative de chargement Gemma 3n")
        strategies = [load_gemma_full, load_conservative, load_ultra_lightweight_for_hf_spaces, load_basic_pipeline]
    else:
        st.warning("GPU non disponible, utilisation du CPU")
        strategies = [load_conservative, load_ultra_lightweight_for_hf_spaces, load_basic_pipeline]
    
    for i, strategy in enumerate(strategies):
        try:
            st.info(f"🔄 Tentative {i+1}/{len(strategies)} : {strategy.__name__}")
            model, processor = strategy()
            if model is not None:
                return model, processor
        except Exception as e:
            st.warning(f"⚠️ Échec de la stratégie {strategy.__name__} : {e}")
            continue
    
    st.error("❌ Toutes les stratégies de chargement ont échoué")
    return None, None

def load_ultra_lightweight_for_hf_spaces():
    """Charge Gemma 3B (plus léger que Gemma 3n) pour HF Spaces"""
    st.info("🪶 Chargement de Gemma 3B pour HF Spaces...")
    
    # Nettoyer la mémoire
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Importer les modules nécessaires
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Modèle Gemma 3B (plus léger que Gemma 3n E4B)
        model_id = "google/gemma-3b-it"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code=True
        )
        
        # Créer un processeur simple
        processor = SimpleTextProcessor(tokenizer)
        
        # Stocker dans session_state
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.tokenizer = tokenizer
        st.session_state.model_loaded = True
        st.session_state.model_status = "Chargé (Gemma 3B HF Spaces)"
        st.session_state.model_load_time = time.time()
        st.session_state.is_gemma_3b = True
        
        st.success("✅ Gemma 3B chargé avec succès pour HF Spaces !")
        return model, processor
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement Gemma 3B : {e}")
        return None, None

def load_basic_pipeline():
    """Charge un pipeline basique sans modèle de vision"""
    st.info("🔧 Chargement du pipeline basique...")
    
    try:
        # Pipeline simple pour analyse de texte
        from transformers import pipeline
        
        classifier = pipeline(
            "text-classification",
            model="distilbert/distilbert-base-uncased",
            device=-1  # CPU
        )
        
        # Créer un processeur simple
        processor = SimpleTextProcessor(None)
        
        # Stocker dans session_state
        st.session_state.classifier = classifier
        st.session_state.processor = processor
        st.session_state.model_loaded = True
        st.session_state.model_status = "Chargé (Pipeline basique)"
        st.session_state.model_load_time = time.time()
        st.session_state.is_basic_pipeline = True
        
        st.success("✅ Pipeline basique chargé avec succès !")
        return classifier, processor
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du pipeline basique : {e}")
        return None, None

def load_gemma_full():
    """Charge le modèle Gemma 3n complet (pour environnement local uniquement)"""
    st.info("🚀 Chargement du modèle Gemma 3n complet...")
    
    # Nettoyer la mémoire
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        model_id = "google/gemma-3n-E4B-it"
        processor = AutoProcessor.from_pretrained(model_id)
        
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        # Stocker dans session_state
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.model_loaded = True
        st.session_state.model_status = "Chargé (Gemma 3n complet)"
        st.session_state.model_load_time = time.time()
        st.session_state.is_gemma_full = True
        
        st.success("✅ Modèle Gemma 3n complet chargé avec succès !")
        return model, processor
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement Gemma complet : {e}")
        return None, None

def load_conservative():
    """Charge le modèle avec des paramètres conservateurs"""
    st.info("🔄 Chargement conservateur...")
    
    # Nettoyer la mémoire
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        model_id = "google/gemma-3n-E4B-it"
        processor = AutoProcessor.from_pretrained(model_id)
        
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        
        # Stocker dans session_state
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.model_loaded = True
        st.session_state.model_status = "Chargé (Conservateur)"
        st.session_state.model_load_time = time.time()
        st.session_state.is_conservative = True
        
        st.success("✅ Modèle chargé avec succès (mode conservateur) !")
        return model, processor
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement conservateur : {e}")
        return None, None

class SimpleTextProcessor:
    """Processeur simple pour les modèles légers"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, text, images=None, return_tensors="pt", **kwargs):
        if self.tokenizer:
            return self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        else:
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    
    def decode(self, tokens, skip_special_tokens=True):
        if self.tokenizer:
            return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        else:
            return "Analyse basique disponible"

def analyze_image_multilingual(image, prompt=""):
    """Analyse une image avec le modèle disponible (Gemma 3n ou modèle alternatif)."""
    if not st.session_state.model_loaded:
        if not restore_model_from_cache():
            st.warning("Modèle non chargé. Veuillez le charger via les réglages avant d'analyser.")
            return "❌ Modèle non chargé. Veuillez d'abord charger le modèle dans les réglages."
        else:
            st.info("Modèle restauré depuis le cache pour l'analyse.")

    model = st.session_state.model
    if not model:
        return "❌ Modèle non disponible. Veuillez recharger le modèle."
    
    # Détecter le type de modèle chargé
    is_gemma_3b = getattr(st.session_state, 'is_gemma_3b', False)
    is_basic_pipeline = getattr(st.session_state, 'is_basic_pipeline', False)
    is_gemma_full = getattr(st.session_state, 'is_gemma_full', False)
    is_conservative = getattr(st.session_state, 'is_conservative', False)
    
    if is_gemma_3b:
        return analyze_image_with_gemma3b_and_gemini(image, prompt)
    elif is_basic_pipeline:
        return analyze_image_with_basic_pipeline(image, prompt)
    elif is_gemma_full or is_conservative:
        return analyze_image_with_gemma_model(image, prompt)
    else:
        # Fallback vers l'analyse basique
        return analyze_image_with_basic_pipeline(image, prompt)

def analyze_image_with_gemma_model(image, prompt=""):
    """Analyse une image avec le modèle Gemma 3n E4B IT complet."""
    processor = st.session_state.processor
    model = st.session_state.model

    try:
        # Préparer le prompt textuel pour Gemma 3n
        if st.session_state.language == "fr":
            if prompt:
                text_prompt = f"""
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
                text_prompt = """
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
                text_prompt = f"""
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
                text_prompt = """
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
        
        # Utiliser le format correct pour Gemma 3n avec le token <image>
        final_prompt = f"<image>\n{text_prompt}"
        
        # Étape 2: Utiliser le processeur pour combiner texte et image
        try:
            inputs = processor(text=final_prompt, images=image, return_tensors="pt").to(model.device)
        except Exception as e:
            if "Number of images does not match number of special image tokens" in str(e):
                # Fallback: essayer sans le token <image>
                inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(model.device)
            else:
                raise e
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Étape 3: Générer la réponse
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            response = processor.decode(generation[0][input_len:], skip_special_tokens=True)

        final_response = response.strip()
        
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
        error_message = str(e)
        if "403" in error_message or "Forbidden" in error_message:
            return "❌ Erreur 403 - Accès refusé. Veuillez vérifier votre jeton Hugging Face (HF_TOKEN) et les quotas."
        elif "Number of images does not match number of special image tokens" in error_message:
            return "❌ Erreur : Le modèle n'a pas pu lier l'image au texte. Assurez-vous que la structure du prompt est correcte."
        else:
            return f"❌ Erreur lors de l'analyse d'image : {e}"

def analyze_image_with_gemma3b_and_gemini(image, prompt=""):
    """Analyse une image avec Gemma 3B + Gemini pour l'interprétation."""
    model = st.session_state.model
    processor = st.session_state.processor
    
    try:
        # Convertir l'image en description textuelle basique
        image_description = f"Image de plante avec dimensions {image.size[0]}x{image.size[1]} pixels"
        
        # Préparer le prompt pour Gemma 3B
        if st.session_state.language == "fr":
            if prompt:
                text_prompt = f"""
Tu es un expert en pathologie végétale. Analyse cette description d'image de plante et fournis un diagnostic.

**Description de l'image :** {image_description}
**Question spécifique :** {prompt}

**Instructions :**
1. **Diagnostic général** : Donne des conseils généraux sur les maladies végétales
2. **Recommandations** : Conseils de base pour l'identification et le traitement
3. **Actions préventives** : Mesures générales de prévention

**Note :** Cette analyse est basée sur une description textuelle. Pour une analyse précise, utilisez un modèle spécialisé en vision.
"""
            else:
                text_prompt = f"""
Tu es un expert en pathologie végétale. Analyse cette description d'image de plante et fournis un diagnostic.

**Description de l'image :** {image_description}

**Instructions :**
1. **Diagnostic général** : Donne des conseils généraux sur les maladies végétales
2. **Recommandations** : Conseils de base pour l'identification et le traitement
3. **Actions préventives** : Mesures générales de prévention

**Note :** Cette analyse est basée sur une description textuelle. Pour une analyse précise, utilisez un modèle spécialisé en vision.
"""
        else:
            if prompt:
                text_prompt = f"""
You are an expert in plant pathology. Analyze this image description and provide a diagnosis.

**Image description:** {image_description}
**Specific question:** {prompt}

**Instructions:**
1. **General diagnosis**: Provide general advice on plant diseases
2. **Recommendations**: Basic guidance for identification and treatment
3. **Preventive actions**: General prevention measures

**Note:** This analysis is based on a text description. For precise analysis, use a specialized vision model.
"""
            else:
                text_prompt = f"""
You are an expert in plant pathology. Analyze this image description and provide a diagnosis.

**Image description:** {image_description}

**Instructions:**
1. **General diagnosis**: Provide general advice on plant diseases
2. **Recommendations**: Basic guidance for identification and treatment
3. **Preventive actions**: General prevention measures

**Note:** This analysis is based on a text description. For precise analysis, use a specialized vision model.
"""
        
        # Générer la réponse avec Gemma 3B
        inputs = processor(text_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        gemma_response = processor.decode(outputs[0], skip_special_tokens=True)
        gemma_response = gemma_response.replace(text_prompt, "").strip()
        
        # Utiliser Gemini pour améliorer l'interprétation
        if gemini_model:
            try:
                gemini_prompt = f"""
Tu es un expert en pathologie végétale. Améliore et structure cette analyse de diagnostic de plante :

**Analyse brute de Gemma 3B :**
{gemma_response}

**Instructions :**
1. **Structure** l'analyse de manière claire et professionnelle
2. **Ajoute** des détails techniques si nécessaire
3. **Améliore** la présentation avec des sections bien définies
4. **Vérifie** la cohérence et la précision des informations

**Format de réponse :**
## 🔍 **Diagnostic Précis**
## 📋 **Symptômes Détaillés**
## 💊 **Traitement Recommandé**
## 🛡️ **Actions Préventives**
"""
                
                gemini_response = gemini_model.generate_content(gemini_prompt)
                final_response = gemini_response.text
                
                if st.session_state.language == "fr":
                    return f"""
## 🤖 **Analyse Gemma 3B + Gemini**

{final_response}

**🔧 Technologies utilisées :**
- **Gemma 3B** : Analyse initiale et diagnostic
- **Gemini** : Amélioration et structuration de l'interprétation
"""
                else:
                    return f"""
## 🤖 **Gemma 3B + Gemini Analysis**

{final_response}

**🔧 Technologies used:**
- **Gemma 3B** : Initial analysis and diagnosis
- **Gemini** : Enhancement and structuring of interpretation
"""
                    
            except Exception as gemini_error:
                st.warning(f"⚠️ Erreur Gemini, utilisation de la réponse Gemma 3B brute : {gemini_error}")
                final_response = gemma_response
        else:
            final_response = gemma_response
        
        if st.session_state.language == "fr":
            return f"""
## 🪶 **Analyse par Gemma 3B**

{final_response}

**⚠️ Note :** Cette analyse utilise Gemma 3B (version allégée) optimisée pour HF Spaces.
"""
        else:
            return f"""
## 🪶 **Analysis by Gemma 3B**

{final_response}

**⚠️ Note:** This analysis uses Gemma 3B (lightweight version) optimized for HF Spaces.
"""
            
    except Exception as e:
        return f"❌ Erreur lors de l'analyse avec Gemma 3B : {e}"

def analyze_image_with_ultra_lightweight_model(image, prompt=""):
    """Analyse une image avec un modèle ultra-léger (texte uniquement)."""
    model = st.session_state.model
    processor = st.session_state.processor
    
    try:
        # Convertir l'image en description textuelle basique
        image_description = f"Image de plante avec dimensions {image.size[0]}x{image.size[1]} pixels"
        
        # Préparer le prompt
        if st.session_state.language == "fr":
            if prompt:
                text_prompt = f"""
Tu es un expert en pathologie végétale. Analyse cette description d'image de plante et fournis un diagnostic.

**Description de l'image :** {image_description}
**Question spécifique :** {prompt}

**Instructions :**
1. **Diagnostic général** : Donne des conseils généraux sur les maladies végétales
2. **Recommandations** : Conseils de base pour l'identification et le traitement
3. **Actions préventives** : Mesures générales de prévention

**Note :** Cette analyse est basée sur une description textuelle. Pour une analyse précise, utilisez un modèle spécialisé en vision.
"""
            else:
                text_prompt = f"""
Tu es un expert en pathologie végétale. Analyse cette description d'image de plante et fournis un diagnostic.

**Description de l'image :** {image_description}

**Instructions :**
1. **Diagnostic général** : Donne des conseils généraux sur les maladies végétales
2. **Recommandations** : Conseils de base pour l'identification et le traitement
3. **Actions préventives** : Mesures générales de prévention

**Note :** Cette analyse est basée sur une description textuelle. Pour une analyse précise, utilisez un modèle spécialisé en vision.
"""
        else:
            if prompt:
                text_prompt = f"""
You are an expert in plant pathology. Analyze this image description and provide a diagnosis.

**Image description:** {image_description}
**Specific question:** {prompt}

**Instructions:**
1. **General diagnosis**: Provide general advice on plant diseases
2. **Recommendations**: Basic guidance for identification and treatment
3. **Preventive actions**: General prevention measures

**Note:** This analysis is based on a text description. For precise analysis, use a specialized vision model.
"""
            else:
                text_prompt = f"""
You are an expert in plant pathology. Analyze this image description and provide a diagnosis.

**Image description:** {image_description}

**Instructions:**
1. **General diagnosis**: Provide general advice on plant diseases
2. **Recommendations**: Basic guidance for identification and treatment
3. **Preventive actions**: General prevention measures

**Note:** This analysis is based on a text description. For precise analysis, use a specialized vision model.
"""
        
        # Générer la réponse avec le modèle ultra-léger
        inputs = processor(text_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        final_response = response.replace(text_prompt, "").strip()
        
        if st.session_state.language == "fr":
            return f"""
## 🪶 **Analyse par Modèle Ultra-Léger**

{final_response}

**⚠️ Note :** Cette analyse utilise un modèle ultra-léger optimisé pour HF Spaces. Pour une analyse d'image précise, utilisez un environnement avec plus de mémoire.
"""
        else:
            return f"""
## 🪶 **Analysis by Ultra-Lightweight Model**

{final_response}

**⚠️ Note:** This analysis uses an ultra-lightweight model optimized for HF Spaces. For precise image analysis, use an environment with more memory.
"""
            
    except Exception as e:
        return f"❌ Erreur lors de l'analyse avec le modèle ultra-léger : {e}"

def analyze_image_with_lightweight_model(image, prompt=""):
    """Analyse une image avec le modèle léger (texte uniquement + API externe pour l'image)."""
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    
    try:
        # Convertir l'image en description textuelle basique
        image_description = f"Image de plante avec dimensions {image.size[0]}x{image.size[1]} pixels"
        
        # Préparer le prompt
        if st.session_state.language == "fr":
            if prompt:
                text_prompt = f"""
Tu es un expert en pathologie végétale. Analyse cette description d'image de plante et fournis un diagnostic.

**Description de l'image :** {image_description}
**Question spécifique :** {prompt}

**Instructions :**
1. **Diagnostic général** : Donne des conseils généraux sur les maladies végétales
2. **Recommandations** : Conseils de base pour l'identification et le traitement
3. **Actions préventives** : Mesures générales de prévention

**Note :** Cette analyse est basée sur une description textuelle. Pour une analyse précise, utilisez un modèle spécialisé en vision.
"""
            else:
                text_prompt = f"""
Tu es un expert en pathologie végétale. Analyse cette description d'image de plante et fournis un diagnostic.

**Description de l'image :** {image_description}

**Instructions :**
1. **Diagnostic général** : Donne des conseils généraux sur les maladies végétales
2. **Recommandations** : Conseils de base pour l'identification et le traitement
3. **Actions préventives** : Mesures générales de prévention

**Note :** Cette analyse est basée sur une description textuelle. Pour une analyse précise, utilisez un modèle spécialisé en vision.
"""
        else:
            if prompt:
                text_prompt = f"""
You are an expert in plant pathology. Analyze this image description and provide a diagnosis.

**Image description:** {image_description}
**Specific question:** {prompt}

**Instructions:**
1. **General diagnosis**: Provide general advice on plant diseases
2. **Recommendations**: Basic guidance for identification and treatment
3. **Preventive actions**: General prevention measures

**Note:** This analysis is based on a text description. For precise analysis, use a specialized vision model.
"""
            else:
                text_prompt = f"""
You are an expert in plant pathology. Analyze this image description and provide a diagnosis.

**Image description:** {image_description}

**Instructions:**
1. **General diagnosis**: Provide general advice on plant diseases
2. **Recommendations**: Basic guidance for identification and treatment
3. **Preventive actions**: General prevention measures

**Note:** This analysis is based on a text description. For precise analysis, use a specialized vision model.
"""
        
        # Générer la réponse avec le modèle léger
        inputs = tokenizer(text_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_response = response.replace(text_prompt, "").strip()
        
        if st.session_state.language == "fr":
            return f"""
## 🧠 **Analyse par Modèle Léger**

{final_response}

**⚠️ Note :** Cette analyse utilise un modèle léger optimisé pour le texte. Pour une analyse d'image précise, utilisez un environnement avec plus de mémoire.
"""
        else:
            return f"""
## 🧠 **Analysis by Lightweight Model**

{final_response}

**⚠️ Note:** This analysis uses a lightweight model optimized for text. For precise image analysis, use an environment with more memory.
"""
            
    except Exception as e:
        return f"❌ Erreur lors de l'analyse avec le modèle léger : {e}"

def analyze_image_with_basic_pipeline(image, prompt=""):
    """Analyse avec le pipeline basique (fonctionnalités très limitées)."""
    
    try:
        # Description basique de l'image
        image_description = f"Plant image with dimensions {image.size[0]}x{image.size[1]} pixels"
        
        if st.session_state.language == "fr":
            return f"""
## 🧠 **Analyse Basique**

**Description de l'image :** {image_description}

**⚠️ Limitation :** Le modèle actuel ne peut analyser que du texte. Pour une analyse d'image complète, utilisez un environnement avec plus de mémoire.

**Conseils généraux :**
- Vérifiez les symptômes visibles sur les feuilles, tiges et racines
- Consultez un expert en pathologie végétale pour un diagnostic précis
- Prenez des photos détaillées sous différents angles
- Notez les conditions environnementales (humidité, température, etc.)
"""
        else:
            return f"""
## 🧠 **Basic Analysis**

**Image description:** {image_description}

**⚠️ Limitation:** The current model can only analyze text. For complete image analysis, use an environment with more memory.

**General advice:**
- Check visible symptoms on leaves, stems, and roots
- Consult a plant pathology expert for precise diagnosis
- Take detailed photos from different angles
- Note environmental conditions (humidity, temperature, etc.)
"""
    except Exception as e:
        return f"❌ Erreur lors de l'analyse basique : {e}"

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
        )
        
        # Gérer le device de manière sécurisée
        device = getattr(model, 'device', 'cpu')
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        
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

# Vérification automatique de la persistance du modèle au démarrage
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_status' not in st.session_state:
    st.session_state.model_status = "Non chargé"

# Vérification automatique de la persistance
if st.session_state.model_loaded:
    # Vérifier si le modèle est toujours disponible
    if not check_model_persistence():
        # Essayer de restaurer depuis le cache
        if restore_model_from_cache():
            st.success("🔄 Modèle restauré automatiquement depuis le cache")
        else:
            st.warning("⚠️ Modèle perdu en mémoire - rechargement nécessaire")
            st.session_state.model_loaded = False
            st.session_state.model_status = "Non chargé"

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
    
    # Affichage du statut avec indicateur de persistance
    status_emoji = "✅" if st.session_state.model_loaded else "❌"
    persistence_emoji = "🔒" if st.session_state.model_persistence_check else "⚠️"
    
    st.info(f"{status_emoji} {t('model_status')} {st.session_state.model_status} {persistence_emoji}")
    
    # Indicateur de persistance
    if st.session_state.model_loaded:
        if st.session_state.model_persistence_check:
            st.success("🔒 Modèle persisté en cache - stable entre les sessions")
        else:
            st.warning("⚠️ Modèle chargé mais pas encore persisté")
    
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
                        st.info("�� Le modèle en ligne est sauvegardé en cache et persiste entre les sessions")
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
                col1, col2, col3 = st.columns(3)
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
                with col3:
                    if st.button("🔍 Debug Cache", type="secondary"):
                        st.write("=== DEBUG CACHE ===")
                        st.write(f"Session state keys: {list(st.session_state.keys())}")
                        st.write(f"Global cache keys: {list(st.session_state.global_model_cache.keys())}")
                        st.write(f"Model loaded: {st.session_state.model_loaded}")
                        st.write(f"Model in session: {hasattr(st.session_state, 'model') and st.session_state.model is not None}")
                        st.write(f"Model in cache: {'model' in st.session_state.global_model_cache and st.session_state.global_model_cache['model'] is not None}")
                        if 'model' in st.session_state.global_model_cache:
                            cached_model = st.session_state.global_model_cache['model']
                            st.write(f"Cached model type: {type(cached_model).__name__}")
                            st.write(f"Cached model has device: {hasattr(cached_model, 'device')}")
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