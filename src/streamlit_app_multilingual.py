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

# --- Configuration de la Page ---
st.set_page_config(
    page_title="AgriLens AI - Diagnostic des Plantes",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Traductions (Utilisation d'un dictionnaire simple) ---
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
    "creator_portfolio": {"fr": "Hugging Face Portfolio: Sidoineko/portfolio", "en": "Hugging Face Portfolio: Sidoineko/portfolio"},
    "competition_title": {"fr": "🏆 Version Compétition Kaggle", "en": "🏆 Kaggle Competition Version"},
    "competition_text": {"fr": "Cette première version d'AgriLens AI a été développée spécifiquement pour participer à la compétition Kaggle. Elle représente notre première production publique et démontre notre expertise en IA appliquée à l'agriculture.", "en": "This first version of AgriLens AI was specifically developed to participate in the Kaggle competition. It represents our first public production and demonstrates our expertise in AI applied to agriculture."},
    "footer": {"fr": "*AgriLens AI - Diagnostic intelligent des plantes avec IA*", "en": "*AgriLens AI - Intelligent plant diagnosis with AI*"}
}

def t(key):
    """Fonction de traduction simple."""
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
    """Vérifie si le modèle est toujours persistant en mémoire et fonctionnel."""
    try:
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            if hasattr(st.session_state.model, 'device'):
                return True
        return False
    except Exception:
        return False

def force_model_persistence():
    """Force la persistance du modèle et du processeur dans le cache global."""
    try:
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            st.session_state.global_model_cache['model'] = st.session_state.model
            st.session_state.global_model_cache['processor'] = st.session_state.processor
            st.session_state.global_model_cache['load_time'] = time.time()
            st.session_state.model_persistence_check = True
            return True
        return False
    except Exception:
        return False

def restore_model_from_cache():
    """Restaure le modèle et le processeur depuis le cache global."""
    try:
        if 'model' in st.session_state.global_model_cache and st.session_state.global_model_cache['model'] is not None:
            st.session_state.model = st.session_state.global_model_cache['model']
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
    """Diagnostique les problèmes potentiels de chargement."""
    issues = []
    if not HfFolder.get_token() and not os.environ.get("HF_TOKEN"):
        issues.append("⚠️ **Jeton Hugging Face (HF_TOKEN) non configuré.** Le téléchargement du modèle pourrait échouer.")
    try:
        import transformers; issues.append(f"✅ Transformers v{transformers.__version__}")
        import torch; issues.append(f"✅ PyTorch v{torch.__version__}")
        if torch.cuda.is_available(): issues.append(f"✅ CUDA disponible : {torch.cuda.get_device_name(0)}")
        else: issues.append("⚠️ CUDA non disponible - utilisation CPU (plus lent)")
    except ImportError as e: issues.append(f"❌ Dépendance manquante : {e}")
    try:
        mem = psutil.virtual_memory()
        issues.append(f"💾 RAM disponible : {mem.available / (1024**3):.2f} GB")
        if mem.available < 4 * 1024**3:
            issues.append("⚠️ RAM insuffisante (< 4GB) - Le chargement risque d'échouer.")
    except ImportError: issues.append("⚠️ Impossible de vérifier la mémoire système")
    return issues

def resize_image_if_needed(image, max_size=(800, 800)):
    """Redimensionne une image PIL si elle dépasse `max_size` tout en conservant les proportions."""
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image, True
    return image, False

# --- Fonctions d'Analyse avec Gemma 3n E4B IT ---
MODEL_ID_HF = "google/gemma-3n-E4B-it"
LOCAL_MODEL_PATH = "D:/Dev/model_gemma"

def load_model_strategy(model_identifier, **kwargs):
    """Charge un modèle et son processeur en utilisant des paramètres spécifiques."""
    try:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        processor = AutoProcessor.from_pretrained(model_identifier, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_identifier,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            token=os.environ.get("HF_TOKEN"),
            **kwargs
        )
        
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.model_loaded = True
        st.session_state.model_status = f"Chargé ({kwargs.get('device_map', 'auto')})"
        st.session_state.model_load_time = time.time()
        
        if kwargs.get('force_persistence'):
            force_model_persistence()
        
        return model, processor
    except Exception as e:
        raise Exception(f"Échec du chargement avec la stratégie: {e}")

def load_model():
    """Charge le modèle Gemma 3n E4B IT avec des stratégies robustes."""
    with st.expander("📊 Diagnostic système", expanded=False):
        for issue in diagnose_loading_issues(): st.markdown(issue)

    is_local = os.path.exists(LOCAL_MODEL_PATH)
    model_id = LOCAL_MODEL_PATH if is_local else MODEL_ID_HF
    st.info(f"Tentative de chargement du modèle depuis : {'Local' if is_local else 'Hugging Face'}...")

    strategies = []
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        st.info(f"Mémoire GPU détectée : {gpu_memory_gb:.1f} GB")
        if gpu_memory_gb >= 10:
            strategies.append(("GPU (float16)", {"device_map": "auto", "torch_dtype": torch.float16}))
        strategies.append(("GPU (8-bit)", {"device_map": "auto", "load_in_8bit": True}))
        strategies.append(("GPU (4-bit)", {"device_map": "auto", "load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}))
    
    strategies.append(("CPU (bfloat16)", {"device_map": "cpu", "torch_dtype": torch.bfloat16}))
    strategies.append(("CPU (float32)", {"device_map": "cpu", "torch_dtype": torch.float32}))

    for name, params in strategies:
        st.info(f"Tentative de chargement via la stratégie : '{name}'...")
        try:
            model, processor = load_model_strategy(model_id, **params)
            if model and processor:
                st.success(f"✅ Modèle chargé avec succès via la stratégie : '{name}'")
                force_model_persistence()
                return model, processor
        except Exception as e:
            st.warning(f"La stratégie '{name}' a échoué : {e}")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    st.error("Toutes les stratégies de chargement du modèle ont échoué.")
    return None, None

# --- NOUVELLE FONCTION CORRIGÉE ---
def analyze_image_multilingual(image, prompt=""):
    """
    Analyse une image avec Gemma 3n E4B IT en utilisant la méthode standard du processeur.
    Cette version est plus robuste et corrige l'erreur de "mismatch".
    """
    if not st.session_state.get('model_loaded', False):
        if not restore_model_from_cache():
            st.warning("Modèle non chargé. Veuillez le charger via les réglages avant d'analyser.")
            return "❌ Modèle Gemma non chargé. Veuillez d'abord charger le modèle dans les réglages."
        else:
            st.info("Modèle restauré depuis le cache pour l'analyse.")

    model, processor = st.session_state.model, st.session_state.processor
    if not model or not processor:
        return "❌ Modèle Gemma non disponible. Veuillez recharger le modèle."

    try:
        # Étape 1: Préparer le prompt textuel qui inclura le jeton <image>
        # Le processeur placera le jeton automatiquement s'il n'est pas déjà présent.
        # Pour être explicite, nous le mettons dans un format de conversation.
        if st.session_state.language == "fr":
            user_prompt = "Analyse cette image de plante et fournis un diagnostic précis, ainsi que les causes, symptômes, traitements et un niveau d'urgence."
            if prompt:
                user_prompt += f"\nQuestion spécifique de l'utilisateur : {prompt}"
        else: # English
            user_prompt = "Analyze this plant image and provide a precise diagnosis, including causes, symptoms, treatments, and an urgency level."
            if prompt:
                user_prompt += f"\nUser's specific question: {prompt}"
        
        # Le modèle Gemma-3n est un modèle "single-turn". On lui donne l'image et la question en une fois.
        # Le format <image>\n{prompt} est le plus efficace.
        final_prompt = f"<image>\n{user_prompt}"

        # Étape 2: Utiliser le processeur pour combiner texte et image
        inputs = processor(text=final_prompt, images=image, return_tensors="pt").to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Étape 3: Générer la réponse
        with torch.inference_mode():
            generation = model.generate(
                **inputs, # Déballer le dictionnaire (input_ids, attention_mask, pixel_values)
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            response = processor.decode(generation[0][input_len:], skip_special_tokens=True)

        final_response = response.strip()
        
        # Étape 4: Formatter le résultat
        header = "## 🧠 **Analyse par Gemma 3n E4B IT**" if st.session_state.language == "fr" else "## 🧠 **Analysis by Gemma 3n E4B IT**"
        return f"{header}\n\n{final_response}"
            
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
            return "❌ Erreur 403 - Accès refusé. Veuillez vérifier votre jeton Hugging Face (HF_TOKEN)."
        elif "Number of images does not match" in error_msg:
            return f"❌ Erreur de correspondance Image/Jeton. Le modèle n'a pas pu associer l'image au texte. Ceci indique un problème interne à la configuration."
        else:
            st.error(f"Une erreur inattendue est survenue : {e}") # Affiche l'erreur pour le débogage
            return f"❌ Erreur lors de l'analyse d'image : {e}"

def analyze_text_multilingual(text):
    """Analyse un texte avec le modèle Gemma 3n E4B IT."""
    if not st.session_state.get('model_loaded', False):
        if not restore_model_from_cache():
            st.warning("Modèle non chargé. Veuillez le charger via les réglages avant d'analyser.")
            return "❌ Modèle non chargé."
        else:
            st.info("Modèle restauré depuis le cache pour l'analyse.")

    model, processor = st.session_state.model, st.session_state.processor
    if not model or not processor: return "❌ Modèle Gemma non disponible."
    
    try:
        if st.session_state.language == "fr":
            prompt_template = f"Tu es un assistant agricole expert. Analyse ce problème de plante : \n\n**Description du problème :**\n{text}\n\n**Instructions :**\n1. **Diagnostic** : Quel est le problème principal ?\n2. **Causes** : Quelles sont les causes possibles ?\n3. **Traitement** : Quelles sont les actions à entreprendre ?\n4. **Prévention** : Comment éviter le problème à l'avenir ?"
        else:
            prompt_template = f"You are an expert agricultural assistant. Analyze this plant problem: \n\n**Problem Description:**\n{text}\n\n**Instructions:**\n1. **Diagnosis**: What is the main problem?\n2. **Causes**: What are the possible causes?\n3. **Treatment**: What actions should be taken?\n4. **Prevention**: How to avoid the problem in the future?"
        
        messages = [{"role": "user", "content": prompt_template}]
        
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        
        input_len = inputs.shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(input_ids=inputs, max_new_tokens=500, do_sample=True, temperature=0.7)
            response = processor.decode(generation[0][input_len:], skip_special_tokens=True)
        
        return response.strip().replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse de texte : {e}"

# --- Interface Principale ---
st.title(t("title"))
st.markdown(t("subtitle"))

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_status' not in st.session_state:
    st.session_state.model_status = "Non chargé"

if not st.session_state.model_loaded and restore_model_from_cache():
    st.success("🔄 Modèle restauré automatiquement depuis le cache.")

with st.sidebar:
    st.header(t("config_title"))
    
    # Language Selection
    lang_map = {"Français": "fr", "English": "en"}
    current_lang_name = "Français" if st.session_state.language == "fr" else "English"
    chosen_lang = st.selectbox("🌐 Langue / Language", options=lang_map.keys(), index=list(lang_map.values()).index(st.session_state.language))
    if lang_map[chosen_lang] != st.session_state.language:
        st.session_state.language = lang_map[chosen_lang]
        st.rerun()

    st.divider()

    # Hugging Face Token Info
    st.subheader("🔑 Jeton Hugging Face")
    if HfFolder.get_token() or os.environ.get("HF_TOKEN"):
        st.success("✅ Jeton HF trouvé et configuré.")
    else:
        st.warning("⚠️ Jeton HF non trouvé. Le téléchargement peut échouer.")
        st.markdown("Il est recommandé de définir la variable d'environnement `HF_TOKEN`. [Obtenir un jeton](https://huggingface.co/settings/tokens)")

    st.divider()

    # Model Control
    st.header("🤖 Modèle IA Gemma 3n")
    if st.session_state.model_loaded and check_model_persistence():
        st.success(f"✅ Modèle chargé ({st.session_state.model_status})")
        if hasattr(st.session_state.model, 'device'): st.write(f"Device : `{st.session_state.model.device}`")
        if st.button("🔄 Recharger le modèle"):
            st.session_state.clear() # Clear everything to be safe
            st.rerun()
    else:
        st.warning("❌ Modèle non chargé.")
        if st.button(t("load_model"), type="primary"):
            with st.spinner("Chargement du modèle en cours... Cette opération peut prendre plusieurs minutes."):
                model, processor = load_model()
            st.rerun()

# --- Onglets Principaux ---
tab1, tab2, tab3, tab4 = st.tabs(t("tabs"))

with tab1: # Image Analysis
    st.header(t("image_analysis_title"))
    st.markdown(t("image_analysis_desc"))
    
    image_source = st.radio("Source de l'image :", ["📁 Upload", "📷 Webcam"], horizontal=True)
    
    image_file = None
    if image_source == "📁 Upload":
        image_file = st.file_uploader(t("choose_image"), type=['png', 'jpg', 'jpeg'])
    else:
        image_file = st.camera_input("Prendre une photo")

    if image_file:
        try:
            image = Image.open(image_file).convert("RGB")
            original_size = image.size
            image, was_resized = resize_image_if_needed(image, max_size=(1024, 1024))
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Image à analyser", use_container_width=True)
                if was_resized: st.info(f"Image redimensionnée de {original_size} à {image.size}")
            
            with col2:
                question = st.text_area("Question spécifique (optionnel) :", placeholder="Ex: Les feuilles ont des taches jaunes, que faire ?", height=100)
                if st.button(t("analyze_button"), disabled=not st.session_state.model_loaded, type="primary"):
                    with st.spinner("🔍 Analyse d'image en cours..."):
                        result = analyze_image_multilingual(image, question)
                    st.markdown(t("analysis_results"))
                    st.markdown("---")
                    st.markdown(result)
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {e}")

with tab2: # Text Analysis
    st.header(t("text_analysis_title"))
    st.markdown(t("text_analysis_desc"))
    
    text_input = st.text_area(t("symptoms_desc"), placeholder="Ex: Mes tomates ont des taches brunes sur les feuilles...", height=150)
    
    if st.button("🧠 Analyser avec l'IA", disabled=not st.session_state.model_loaded, type="primary", key="text_analyze_btn"):
        if not text_input.strip():
            st.error("❌ Veuillez saisir une description des symptômes.")
        else:
            with st.spinner("🔍 Analyse de texte en cours..."):
                result = analyze_text_multilingual(text_input)
            st.markdown(t("analysis_results"))
            st.markdown("---")
            st.markdown(result)

with tab3: # Manual
    st.header(t("manual_title"))
    # (Le contenu du manuel reste le même que dans votre code original)
    manual_content = {
        "fr": """...""", "en": """...""" # Remplir avec votre contenu
    }
    st.markdown(manual_content.get(st.session_state.language, manual_content["en"]))


with tab4: # About
    st.header(t("about_title"))
    # (Le contenu de la section "À propos" reste le même que dans votre code original)
    st.markdown(f"### {t('creator_title')}")
    st.markdown(f"{t('creator_name')}")
    # ... etc.

# --- Pied de page ---
st.markdown("---")
st.markdown(t("footer"))