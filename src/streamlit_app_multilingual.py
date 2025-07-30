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
                device = st.session_state.model.device
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
            st.session_state.global_model_cache['model_type'] = type(st.session_state.model).__name__
            st.session_state.global_model_cache['processor_type'] = type(st.session_state.processor).__name__
            
            if hasattr(st.session_state.model, 'device'):
                st.session_state.global_model_cache['device'] = st.session_state.model.device

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
    """Charge le modèle Gemma 3n E4B IT (local ou Hugging Face) avec des stratégies robustes."""
    try:
        issues = diagnose_loading_issues()
        with st.expander("📊 Diagnostic système", expanded=False):
            for issue in issues:
                st.markdown(issue)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        is_local = os.path.exists(LOCAL_MODEL_PATH)
        strategies_to_try = []

        if is_local:
            strategies_to_try.append(("Local (ultra-conservateur CPU)", lambda: load_model_strategy(LOCAL_MODEL_PATH, device_map="cpu", torch_dtype=torch.bfloat16, quantization=None, force_persistence=True)))
            strategies_to_try.append(("Local (conservateur CPU)", lambda: load_model_strategy(LOCAL_MODEL_PATH, device_map="cpu", torch_dtype=torch.bfloat16, quantization=None, force_persistence=True)))
        else:
            st.info("Modèle local non trouvé. Tentative de chargement depuis Hugging Face...")
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.info(f"Mémoire GPU disponible : {gpu_memory:.1f} GB")
                
                if gpu_memory >= 10:
                    strategies_to_try.append(("Hugging Face (float16)", lambda: load_model_strategy(MODEL_ID_HF, device_map="auto", torch_dtype=torch.float16, quantization=None)))
                if gpu_memory >= 8:
                    strategies_to_try.append(("Hugging Face (8-bit quantization)", lambda: load_model_strategy(MODEL_ID_HF, device_map="auto", torch_dtype=torch.float16, quantization="8bit")))
                if gpu_memory >= 6:
                    strategies_to_try.append(("Hugging Face (4-bit quantization)", lambda: load_model_strategy(MODEL_ID_HF, device_map="auto", torch_dtype=torch.float16, quantization="4bit")))
                
                if gpu_memory < 6:
                     st.warning("Mémoire GPU limitée. L'utilisation du CPU sera probablement plus stable.")

            strategies_to_try.append(("Hugging Face (conservative CPU)", lambda: load_model_strategy(MODEL_ID_HF, device_map="cpu", torch_dtype=torch.float32, quantization=None)))
            strategies_to_try.append(("Hugging Face (ultra-conservative CPU)", lambda: load_model_strategy(MODEL_ID_HF, device_map="cpu", torch_dtype=torch.float32, quantization=None)))

        for i, (name, strategy_func) in enumerate(strategies_to_try):
            st.info(f"Tentative {i+1}/{len(strategies_to_try)} : Chargement via '{name}'...")
            try:
                model, processor = strategy_func()
                if model and processor:
                    st.success(f"✅ Modèle chargé avec succès via la stratégie : '{name}'")
                    return model, processor
            except Exception as e:
                error_msg = str(e)
                st.warning(f"La stratégie '{name}' a échoué : {error_msg}")
                if "disk_offload" in error_msg.lower() or "out of memory" in error_msg.lower():
                    st.warning("Problème de mémoire ou de disk_offload. Tentative suivante...")
                elif "403" in error_msg or "Forbidden" in error_msg:
                    st.error(f"❌ Erreur d'accès Hugging Face (403) avec la stratégie '{name}'. Vérifiez votre HF_TOKEN.")
                    return None, None
                else:
                    st.warning("Tentative suivante...")
                
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue
        
        st.error("Toutes les stratégies de chargement du modèle ont échoué.")
        return None, None

    except ImportError as e:
        st.error(f"❌ Erreur de dépendance : . Installez avec `pip install transformers torch accelerate bitsandbytes`.")
        return None, None
    except Exception as e:
        st.error(f"❌ Une erreur générale s'est produite lors du chargement du modèle : ")
        return None, None

def analyze_image_multilingual(image, prompt=""):
    """
    Analyse une image avec Gemma 3n E4B IT pour un diagnostic précis en utilisant le format chat.
    Cette version tente de contourner le bug de comparaison float/int en forçant l'utilisation
    d'input_ids pour la génération, comme suggéré par le problème #2751.
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
        # Préparer le prompt textuel et visuel pour Gemma 3n en utilisant le format chat
        if st.session_state.language == "fr":
            user_instruction = f"Analyse cette image de plante et fournis un diagnostic précis. Question spécifique : {prompt}" if prompt else "Analyse cette image de plante et fournis un diagnostic précis."
            system_message = "Tu es un expert en pathologie végétale. Réponds de manière structurée et précise, en incluant diagnostic, causes, symptômes, traitement et urgence."
        else: # English
            user_instruction = f"Analyze this plant image and provide a precise diagnosis. Specific question: {prompt}" if prompt else "Analyze this plant image and provide a precise diagnosis."
            system_message = "You are an expert in plant pathology. Respond in a structured and precise manner, including diagnosis, causes, symptoms, treatment, and urgency."
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_instruction}
            ]}
        ]
        
        # Utiliser processor.apply_chat_template pour convertir le format conversationnel en tenseurs
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
            # *** LA MODIFICATION CRUCIALE POUR LE BUG #2751 ***
            # Passer explicitement 'input_ids' et 'attention_mask' et s'assurer qu'ils sont sur le bon device.
            # Ne pas déballer le dictionnaire entier 'inputs' directement pour éviter des problèmes
            # avec d'autres clés comme 'pixel_values' qui pourraient déclencher le bug.
            generation = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                pixel_values=inputs["pixel_values"].to(device) if "pixel_values" in inputs else None, # Passer pixel_values si présent
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
            return f"❌ Erreur lors de l'analyse d'image : {e}"

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
st.title(t("title"))
st.markdown(t("subtitle"))

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_status' not in st.session_state:
    st.session_state.model_status = "Non chargé"

if not st.session_state.model_loaded:
    if restore_model_from_cache():
        st.success("🔄 Modèle restauré automatiquement depuis le cache au démarrage.")
    else:
        st.info("💡 Cliquez sur 'Charger le modèle' dans les réglages pour commencer.")

with st.sidebar:
    st.header(t("config_title"))
    
    st.subheader("🌐 Langue / Language")
    language_options = ["Français", "English"]
    if 'language' not in st.session_state:
        st.session_state.language = 'fr'
    current_lang_index = 0 if st.session_state.language == "fr" else 1
    
    language_choice = st.selectbox(
        "Sélectionnez votre langue :",
        language_options,
        index=current_lang_index,
        help="Change la langue de l'interface et des réponses de l'IA."
    )
    if st.session_state.language != ("fr" if language_choice == "Français" else "en"):
        st.session_state.language = "fr" if language_choice == "Français" else "en"
        st.rerun()

    st.divider()

    st.subheader("🔑 Jeton Hugging Face")
    hf_token_found = HfFolder.get_token() or os.environ.get("HF_TOKEN")
    if hf_token_found:
        st.success("✅ Jeton HF trouvé et configuré.")
    else:
        st.warning("⚠️ Jeton HF non trouvé.")
    st.info("Il est recommandé de définir la variable d'environnement `HF_TOKEN` avec votre jeton personnel Hugging Face pour éviter les erreurs d'accès (403).")
    st.markdown("[Obtenir un jeton HF](https://huggingface.co/settings/tokens)")

    st.divider()

    st.header("🤖 Modèle IA Gemma 3n")
    if st.session_state.model_loaded and check_model_persistence():
        st.success(f"✅ Modèle chargé ({st.session_state.model_status})")
        if st.session_state.model_load_time:
            load_time_str = time.strftime('%H:%M:%S', time.localtime(st.session_state.model_load_time))
            st.write(f"Heure de chargement : ")
        if hasattr(st.session_state.model, 'device'):
            st.write(f"Device utilisé : `{st.session_state.model.device}`")
        
        col1_btn, col2_btn = st.columns(2)
        with col1_btn:
            if st.button("🔄 Recharger le modèle", type="secondary"):
                st.session_state.model_loaded = False
                st.session_state.model = None
                st.session_state.processor = None
                st.session_state.global_model_cache.clear()
                st.session_state.model_persistence_check = False
                st.rerun()
        with col2_btn:
            if st.button("💾 Forcer Persistance", type="secondary"):
                if force_model_persistence():
                    st.success("Persistance forcée avec succès.")
                else:
                    st.error("Échec de la persistance.")
                st.rerun()
    else:
        st.warning("❌ Modèle non chargé.")
        if st.button(t("load_model"), type="primary"):
            with st.spinner("Chargement du modèle en cours..."):
                model, processor = load_model()
                if model and processor:
                    st.success("✅ Modèle chargé avec succès !")
                else:
                    st.error("❌ Échec du chargement du modèle.")
            st.rerun()

    st.divider()
    st.subheader("💾 Persistance du Modèle")
    if st.session_state.model_loaded and st.session_state.model_persistence_check:
        st.success("✅ Modèle chargé et persistant en cache.")
    elif st.session_state.model_loaded:
        st.warning("⚠️ Modèle chargé mais non persistant. Cliquez sur 'Forcer Persistance'.")
    else:
        st.warning("⚠️ Modèle non chargé.")


# --- Onglets Principaux ---
tab1, tab2, tab3, tab4 = st.tabs(t("tabs"))

with tab1:
    st.header(t("image_analysis_title"))
    st.markdown(t("image_analysis_desc"))
    
    capture_option = st.radio(
        "Choisissez votre méthode :",
        ["📁 Upload d'image", "📷 Capture par webcam"],
        horizontal=True,
        key="image_capture_method"
    )
    
    uploaded_file = None
    captured_image = None
    
    if capture_option == "📁 Upload d'image":
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
        st.markdown("**📷 Capture d'image par webcam**")
        st.info("💡 Positionnez votre plante malade devant la webcam et cliquez sur 'Prendre une photo'. Assurez-vous d'un bon éclairage.")
        captured_image = st.camera_input("Prendre une photo de la plante", key="webcam_capture")
    
    image = None
    image_source = None
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_source = "upload"
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement de l'image uploadée : ")
            st.info("💡 Essayez avec une image différente ou un format différent (PNG, JPG, JPEG).")
    elif captured_image is not None:
        try:
            image = Image.open(captured_image)
            image_source = "webcam"
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement de l'image capturée : ")
            st.info("💡 Essayez de reprendre la photo.")
    
    if image is not None:
        try:
            original_size = image.size
            image, was_resized = resize_image_if_needed(image, max_size=(800, 800))
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption=f"Image ()" if image_source else "Image", use_container_width=True)
                if was_resized:
                    st.warning(f"⚠️ L'image a été redimensionnée de  à {image.size} pour optimiser le traitement.")
            
            with col2:
                st.markdown("**Informations de l'image :**")
                st.write(f"• Format : {image.format}")
                st.write(f"• Taille originale : {original_size[0]}x{original_size[1]} pixels")
                st.write(f"• Taille actuelle : {image.size[0]}x{image.size[1]} pixels")
                st.write(f"• Mode : {image.mode}")
            
            question = st.text_area(
                "Question spécifique (optionnel) :",
                placeholder="Ex: Les feuilles ont des taches jaunes, que faire ?",
                height=100
            )
            
            if st.button(t("analyze_button"), disabled=not st.session_state.model_loaded, type="primary"):
                if not st.session_state.model_loaded:
                    st.error("❌ Modèle non chargé. Veuillez le charger dans les réglages.")
                else:
                    with st.spinner("🔍 Analyse d'image en cours..."):
                        result = analyze_image_multilingual(image, question)
                    
                    st.markdown(t("analysis_results"))
                    st.markdown("---")
                    st.markdown(result)
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : ")

with tab2:
    st.header(t("text_analysis_title"))
    st.markdown(t("text_analysis_desc"))
    
    text_input = st.text_area(
        t("symptoms_desc"),
        placeholder="Ex: Mes tomates ont des taches brunes sur les feuilles et les fruits, une poudre blanche sur les tiges...",
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

with tab3:
    st.header(t("manual_title"))
    
    manual_content = {
        "fr": """
        ### 🚀 **Démarrage Rapide**
        1.  **Charger le modèle** : Cliquez sur 'Charger le modèle' dans les réglages (sidebar).
        2.  **Choisir le mode** : Allez à l'onglet '📸 Analyse d'Image' ou '💬 Analyse de Texte'.
        3.  **Soumettre votre demande** : Upload d'image, capture webcam, ou description textuelle.
        4.  **Obtenir le diagnostic** : Lisez les résultats avec recommandations.
        
        ### 📸 **Analyse d'Image**
        *   **Formats acceptés** : PNG, JPG, JPEG.
        *   **Qualité** : Privilégiez des images claires, bien éclairées, avec le problème bien visible.
        *   **Redimensionnement** : Les images trop grandes sont automatiquement redimensionnées pour optimiser le traitement.
        
        ### 💬 **Analyse de Texte**
        *   **Soyez précis** : Décrivez les symptômes, le type de plante, les conditions de culture, et les actions déjà tentées. Plus la description est détaillée, plus le diagnostic sera pertinent.
        
        ### 🔍 **Interprétation des Résultats**
        *   Les résultats incluent un diagnostic potentiel, les causes probables, des recommandations de traitement et des conseils de prévention.
        *   Ces informations sont basées sur l'IA et doivent être considérées comme un guide. Consultez un expert pour des cas critiques.
        
        ### 💡 **Bonnes Pratiques**
        *   **Images multiples** : Si possible, prenez des photos sous différents angles.
        *   **Éclairage** : La lumière naturelle est idéale.
        *   **Focus** : Assurez-vous que la zone affectée est nette et bien visible.
        
        ### 💾 **Persistance du Modèle**
        *   Une fois chargé, le modèle est sauvegardé dans le cache de l'application pour un chargement plus rapide lors des prochaines utilisations sur le même environnement.
        *   Vous pouvez le recharger manuellement si nécessaire.
        
        ### 🔒 **Jeton Hugging Face (HF_TOKEN)**
        *   Pour garantir la stabilité et la performance lors du téléchargement de modèles depuis Hugging Face, il est fortement recommandé de définir la variable d'environnement `HF_TOKEN`.
        *   Créez un jeton de lecture sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) et définissez-le dans votre environnement avant de lancer l'application.
        """,
        "en": """
        ### 🚀 **Quick Start**
        1.  **Load the model** : Click 'Load Model' in the settings (sidebar).
        2.  **Choose mode** : Navigate to '📸 Image Analysis' or '💬 Text Analysis' tab.
        3.  **Submit your request** : Upload an image, capture via webcam, or provide a text description.
        4.  **Get diagnosis** : Read the results with recommendations.
        
        ### 📸 **Image Analysis**
        *   **Accepted formats** : PNG, JPG, JPEG.
        *   **Quality** : Prefer clear, well-lit images with the problem clearly visible.
        *   **Resizing** : Oversized images are automatically resized for processing optimization.
        
        ### 💬 **Text Analysis**
        *   **Be specific** : Describe symptoms, plant type, growing conditions, and actions already taken. More detail leads to better accuracy.
        
        ### 🔍 **Result Interpretation**
        *   Results include a potential diagnosis, likely causes, treatment recommendations, and preventive advice.
        *   This AI-driven information is for guidance only. Consult a qualified expert for critical cases.
        
        ### 💡 **Best Practices**
        *   **Multiple images** : If possible, take photos from different angles.
        *   **Lighting** : Natural light is ideal.
        *   **Focus** : Ensure the affected area is sharp and clearly visible.
        
        ### 💾 **Model Persistence**
        *   Once loaded, the model is cached for faster loading in future sessions on the same environment.
        *   You can manually reload it if needed.
        
        ### 🔒 **Hugging Face Token (HF_TOKEN)**
        *   To ensure stability and performance when downloading models from Hugging Face, it's highly recommended to set the environment variable `HF_TOKEN`.
        *   Create a read token on [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and set it in your environment before launching the app.
        """
    }
    st.markdown(manual_content[st.session_state.language])

with tab4:
    st.header(t("about_title"))
    
    st.markdown("### 🌱 Notre Mission / Our Mission")
    st.markdown("AgriLens AI est une application de diagnostic des maladies de plantes utilisant l'intelligence artificielle pour aider les agriculteurs à identifier et traiter les problèmes de leurs cultures.")
    
    st.markdown("### 🚀 Fonctionnalités / Features")
    st.markdown("""
    • **Analyse d'images** : Diagnostic visuel des maladies
    • **Analyse de texte** : Conseils basés sur les descriptions
    • **Recommandations pratiques** : Actions concrètes à entreprendre
    • **Interface optimisée** : Pour une utilisation sur divers appareils
    • **Support multilingue** : Français et Anglais
    """)
    
    st.markdown("### 🔧 Technologie / Technology")
    
    is_local = os.path.exists(LOCAL_MODEL_PATH)
    
    if is_local:
        st.markdown(f"""
        • **Modèle** : Gemma 3n E4B IT (Local - {LOCAL_MODEL_PATH})
        • **Framework** : Streamlit
        • **Déploiement** : Local
        """)
    else:
        st.markdown("""
        • **Modèle** : Gemma 3n E4B IT (Hugging Face - en ligne)
        • **Framework** : Streamlit
        • **Déploiement** : Hugging Face Spaces
        """)
    
    st.markdown(f"### {t('creator_title')}")
    st.markdown(f"{t('creator_name')}")
    st.markdown(f"📍 {t('creator_location')}")
    st.markdown(f"📞 {t('creator_phone')}")
    st.markdown(f"📧 {t('creator_email')}")
    st.markdown(f"🔗 [{t('creator_linkedin')}](https://{t('creator_linkedin')})")
    st.markdown(f"📁 {t('creator_portfolio')}")
    
    st.markdown(f"### {t('competition_title')}")
    st.markdown(t("competition_text"))
    
    st.markdown("### ⚠️ Avertissement / Warning")
    st.markdown("Les résultats fournis par l'IA sont à titre indicatif uniquement et ne remplacent pas l'avis d'un expert agricole qualifié.")
    
    st.markdown("### 📞 Support")
    st.markdown("Pour toute question ou problème, consultez la documentation ou contactez le créateur.")

# --- Pied de page ---
st.markdown("---")
st.markdown(t("footer"))