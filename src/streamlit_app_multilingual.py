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
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import HfFolder, hf_hub_download, snapshot_download
from functools import lru_cache

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="AgriLens AI - Diagnostic des Plantes",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
    "plant_type_prompt": {"fr": "Le type de plante est : **MANIOC**.", "en": "The plant type is: **MANIOC**."},
    "image_prompt_details": {
        "fr": "J'observe les symptômes suivants sur la feuille :",
        "en": "I observe the following symptoms on the leaf:"
    },
    "ask_plant_identification": {
        "fr": "Je veux savoir si c'est une maladie courante du manioc (par exemple, mosaïque, bactériose) ou potentiellement une maladie d'un autre type de plante.",
        "en": "I want to know if it's a common manioc disease (e.g., mosaic, bacterial blight) or potentially a disease of another plant type."
    },
    "analysis_structure": {
        "fr": """
    Structure ta réponse en :
    1.  Identification probable (type de plante et maladie).
    2.  Description des symptômes observés.
    3.  Causes possibles.
    4.  Recommandations de traitement.
    5.  Conseils de prévention.
    """,
        "en": """
    Structure your response into:
    1.  Probable identification (plant type and disease).
    2.  Description of observed symptoms.
    3.  Possible causes.
    4.  Treatment recommendations.
    5.  Prevention tips.
    """
    },
    "text_prompt_details": {
        "fr": "Tu es un assistant agricole expert spécialisé dans les cultures tropicales, comme le MANIOC et le MANGUIER. Analyse ce problème de plante en te concentrant sur les symptômes décrits pour le MANIOC : \n\n**Description du problème :**\n",
        "en": "You are an expert agricultural assistant, specialized in tropical crops like MANIOC and MANGO. Analyze this plant problem, focusing on the described symptoms for MANIOC: \n\n**Problem Description:**\n"
    },
    "text_prompt_instructions": {
        "fr": "\n\n**Instructions spécifiques pour le diagnostic :**\n1. **Identifier la plante** : Est-ce du manioc ou autre chose ? Sois très prudent sur cette identification.\n2. **Identifier la maladie** : Décris les symptômes spécifiques observés sur le manioc (taches, couleur, forme, texture, progression).\n3. **Causes possibles** : Qu'est-ce qui pourrait causer ces symptômes sur du MANIOC ?\n4. **Traitement** : Quelles actions prendre pour soigner le MANIOC ?\n5. **Prévention** : Comment éviter ces problèmes sur le MANIOC à l'avenir ?",
        "en": "\n\n**Specific Diagnosis Instructions:**\n1. **Identify the plant**: Is it MANIOC or something else? Be very careful with this identification.\n2. **Identify the disease**: Describe the specific symptoms observed on MANIOC (spots, color, shape, texture, progression).\n3. **Possible causes**: What could cause these symptoms on MANIOC?\n4. **Treatment**: What actions should be taken to treat the MANIOC?\n5. **Prevention**: How to avoid these problems on MANIOC in the future?"
    }
}

def t(key):
    """Fonction de traduction simple."""
    if 'language' not in st.session_state:
        st.session_state.language = 'fr'
    lang = st.session_state.language
    return TRANSLATIONS.get(key, {}).get(lang, key)

# --- INITIALISATION DE LA LANGUE ET DES CONSTATATIONS GLOBALES ---
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

# --- FONCTIONS UTILITAIRES SYSTÈME ---
def get_device():
    """Détermine le meilleur device disponible (GPU ou CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

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
    """Redimensionne une image PIL si elle dépasse `max_size` tout en conservant les proportions."""
    width, height = image.size
    if width > max_size[0] or height > max_size[1]:
        ratio = min(max_size[0] / width, max_size[1] / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image, True
    return image, False

def afficher_ram_disponible():
    """Affiche l'utilisation de la RAM."""
    try:
        mem = psutil.virtual_memory()
        st.info(f"💾 RAM disponible : {mem.available // (1024**3)} GB")
        if mem.available < 4 * 1024**3: # Seuil de 4GB RAM
            st.warning("⚠️ Moins de 4GB de RAM disponible, le chargement du modèle risque d'échouer !")
    except ImportError:
        st.warning("⚠️ Impossible de vérifier la RAM système.")

# --- CHARGEMENT DU MODÈLE AVEC CACHING ---
# Modèle principal à utiliser
MODEL_ID_HF = "google/gemma-3n-e2b-it" # ID du modèle Gemma 3n e2b it

@st.cache_resource(show_spinner=True) # Cache la ressource (modèle) entre les exécutions
def load_ai_model(model_identifier, device_map="auto", torch_dtype=torch.float16, quantization=None):
    """
    Charge le modèle et son processeur avec les configurations spécifiées.
    Retourne le modèle et le processeur, ou lève une exception en cas d'échec.
    """
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        st.info(f"Tentative de chargement du modèle : `{model_identifier}`")
        
        # --- Configuration des arguments pour le chargement ---
        common_args = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True, # Aide à réduire l'utilisation CPU lors du chargement
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "token": os.environ.get("HF_TOKEN") or HfFolder.get_token() # Récupère le token depuis l'env ou le cache HF
        }
        
        # Configuration de la quantisation
        if quantization == "4bit":
            common_args.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16 # Ou torch.bfloat16 si supporté
            })
        elif quantization == "8bit":
            common_args.update({"load_in_8bit": True})
        
        # --- Chargement du processeur ---
        st.info("Chargement du processeur...")
        processor = AutoProcessor.from_pretrained(model_identifier, **common_args)
        
        # --- Chargement du modèle ---
        st.info("Chargement du modèle...")
        # Utiliser AutoModelForCausalLM car Gemma est un modèle causal
        model = AutoModelForCausalLM.from_pretrained(model_identifier, **common_args)
        
        st.success(f"✅ Modèle `{model_identifier}` chargé avec succès sur device `{device_map}`.")
        return model, processor

    except ImportError as e:
        raise ImportError(f"Erreur de dépendance : {e}. Assurez-vous que `transformers`, `torch`, `accelerate`, et `bitsandbytes` sont installés.")
    except ValueError as e:
        if "403" in str(e) or "Forbidden" in str(e):
            raise ValueError("❌ Erreur d'accès Hugging Face (403). Vérifiez votre jeton Hugging Face (HF_TOKEN). Il doit être défini et valide.")
        else:
            raise ValueError(f"Erreur de configuration du modèle : {e}")
    except Exception as e:
        raise RuntimeError(f"Une erreur est survenue lors du chargement du modèle : {e}")

def get_model_and_processor():
    """
    Stratégie de chargement du modèle Gemma 3n e2b it.
    Essaie différentes configurations pour s'adapter aux ressources disponibles.
    """
    # --- Diagnostic initial ---
    issues = diagnose_loading_issues()
    with st.expander("📊 Diagnostic système", expanded=False):
        for issue in issues:
            st.markdown(issue)

    # --- Stratégies de chargement ---
    strategies = []
    device = get_device()
    
    # Priorité aux stratégies GPU si disponibles
    if device == "cuda":
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.info(f"Mémoire GPU disponible : {gpu_memory_gb:.1f} GB")
        
        # Stratégies GPU par ordre de consommation mémoire décroissante
        if gpu_memory_gb >= 12: # Idéal pour float16
            strategies.append({"name": "GPU (float16)", "config": {"device_map": "auto", "torch_dtype": torch.float16, "quantization": None}})
        if gpu_memory_gb >= 10: # Peut fonctionner avec float16
            strategies.append({"name": "GPU (float16)", "config": {"device_map": "auto", "torch_dtype": torch.float16, "quantization": None}})
        if gpu_memory_gb >= 8: # Recommandé pour 8-bit quant.
            strategies.append({"name": "GPU (8-bit quant.)", "config": {"device_map": "auto", "torch_dtype": torch.float16, "quantization": "8bit"}})
        if gpu_memory_gb >= 6: # Minimum pour 4-bit quant.
            strategies.append({"name": "GPU (4-bit quant.)", "config": {"device_map": "auto", "torch_dtype": torch.float16, "quantization": "4bit"}})
        
        if gpu_memory_gb < 6:
             st.warning("Mémoire GPU limitée (<6GB). Le chargement sur CPU est recommandé.")

    # Stratégies CPU (plus lentes, mais plus robustes sur peu de ressources)
    strategies.append({"name": "CPU (bfloat16)", "config": {"device_map": "cpu", "torch_dtype": torch.bfloat16, "quantization": None}})
    strategies.append({"name": "CPU (float32)", "config": {"device_map": "cpu", "torch_dtype": torch.float32, "quantization": None}})
    
    # --- Tentative de chargement via les stratégies ---
    for strat in strategies:
        st.info(f"Essai : {strat['name']}...")
        try:
            model, processor = load_ai_model(
                MODEL_ID_HF,
                device_map=strat["config"]["device_map"],
                torch_dtype=strat["config"]["torch_dtype"],
                quantization=strat["config"]["quantization"]
            )
            if model and processor:
                st.success(f"Succès avec la stratégie : {strat['name']}")
                return model, processor
        except Exception as e:
            st.warning(f"Échec avec '{strat['name']}' : {e}")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            time.sleep(1)

    raise RuntimeError("Toutes les stratégies de chargement du modèle ont échoué.")

# --- FONCTIONS D'ANALYSE ---
def analyze_image_multilingual(image, user_details_prompt=""):
    """
    Analyse une image avec Gemma 3n e2b it pour un diagnostic précis en utilisant le format chat.
    Intègre les détails de la plante et des symptômes fournis par l'utilisateur.
    """
    model, processor = st.session_state.model, st.session_state.processor
    if not model or not processor:
        return "❌ Modèle IA non chargé. Veuillez charger le modèle dans les réglages."

    try:
        # Construire le prompt basé sur les détails fournis par l'utilisateur
        plant_identification_part = f"Le type de plante est : **MANIOC**." # Par défaut, mais sera surchargé par user_details_prompt si présent.
        symptoms_description_part = ""
        plant_identification_instruction = ""
        
        if user_details_prompt.strip():
            # Essayer d'extraire des informations clés du prompt utilisateur
            # C'est une approche simplifiée. Une analyse NLP plus poussée serait plus robuste.
            if "manioc" in user_details_prompt.lower():
                plant_identification_part = "Le type de plante est : **MANIOC**."
            elif "manguier" in user_details_prompt.lower():
                plant_identification_part = "Le type de plante est : **MANGO**."
            else:
                 plant_identification_part = "Le type de plante n'est pas clairement spécifié, mais le diagnostic doit être fait avec prudence."
                 
            symptoms_description_part = f"J'observe les symptômes suivants : {user_details_prompt}"
            plant_identification_instruction = t("ask_plant_identification")
        else:
            # Prompt par défaut si l'utilisateur ne fournit pas de détails
            plant_identification_part = "Le type de plante n'est pas clairement spécifié. Soyez prudent lors de l'identification."
            symptoms_description_part = "Je n'ai pas de détails spécifiques sur les symptômes."
            plant_identification_instruction = "Essayez d'identifier le type de plante et la maladie."
        
        # Assembler le prompt final pour le modèle
        if st.session_state.language == "fr":
            system_message = "Tu es un expert en pathologie végétale, spécialisé dans les cultures tropicales comme le MANIOC et le MANGUIER. Réponds de manière structurée et précise, en incluant diagnostic, causes, symptômes, traitement et urgence."
            user_instruction = f"{plant_identification_part} {symptoms_description_part} {plant_identification_instruction} {t('analysis_structure')['fr']}"
        else: # English
            system_message = "You are an expert in plant pathology, specialized in tropical crops like MANIOC and MANGO. Respond in a structured and precise manner, including diagnosis, causes, symptoms, treatment, and urgency."
            user_instruction = f"{plant_identification_part} {symptoms_description_part} {plant_identification_instruction} {t('analysis_structure')['en']}"

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": image}, # L'image est transmise directement
                {"type": "text", "text": user_instruction}
            ]}
        ]
        
        # Utiliser apply_chat_template pour convertir le format conversationnel en tenseurs
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        device = getattr(model, 'device', 'cpu')
        # Déplacer les tenseurs sur le bon device
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, # Déballer le dictionnaire des inputs
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            response = processor.decode(generation[0][input_len:], skip_special_tokens=True)

        final_response = response.strip()
        # Nettoyage des tokens de contrôle si présents
        final_response = final_response.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()
        
        # Formatage de la réponse pour l'affichage
        return f"## 🧠 **Analyse par Gemma 3n e2b it**\n\n{final_response}"
            
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
            return "❌ Erreur 403 - Accès refusé. Vérifiez votre jeton Hugging Face (HF_TOKEN) et les quotas."
        elif "Number of images does not match number of special image tokens" in error_msg:
            return "❌ Erreur : Le modèle n'a pas pu associer l'image au texte. Ceci est un bug connu (#2751) lié aux versions de Transformers/Gemma. Essayez de mettre à jour vos bibliothèques (`transformers`, `torch`, `accelerate`)."
        else:
            return f"❌ Erreur lors de l'analyse d'image : {e}"

def analyze_text_multilingual(text):
    """Analyse un texte avec le modèle Gemma 3n e2b it."""
    model, processor = st.session_state.model, st.session_state.processor
    if not model or not processor:
        return "❌ Modèle IA non chargé. Veuillez charger le modèle dans les réglages."
        
    try:
        # Construction du prompt selon la langue
        if st.session_state.language == "fr":
            prompt_template = f"{t('text_prompt_details')['fr']} {text}{t('text_prompt_instructions')['fr']}"
        else: # English
            prompt_template = f"{t('text_prompt_details')['en']} {text}{t('text_prompt_instructions')['en']}"
        
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
                repetition_penalty=1.1
            )
            
            response = processor.decode(generation[0][input_len:], skip_special_tokens=True)
        
        cleaned_response = response.strip()
        cleaned_response = cleaned_response.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()

        return cleaned_response
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse de texte : {e}"

# --- INTERFACE UTILISATEUR STREAMLIT ---
st.title(t("title"))
st.markdown(t("subtitle"))

# --- INITIALISATION ET GESTION DU MODÈLE ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_status' not in st.session_state:
    st.session_state.model_status = t("not_loaded")

# Tentative de chargement automatique au démarrage si le modèle n'est pas déjà chargé
if not st.session_state.model_loaded:
    try:
        model, processor = get_model_and_processor()
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.model_loaded = True
        st.session_state.model_status = t("loaded")
        st.success("✅ Modèle chargé avec succès au démarrage.")
    except Exception as e:
        st.session_state.model_status = f"{t('error')} : {e}"
        st.error(f"❌ Échec du chargement automatique du modèle : {e}")

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.header(t("config_title"))
    
    # Sélecteur de langue
    st.subheader("🌐 Langue / Language")
    language_options = ["Français", "English"]
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

    # Configuration du jeton Hugging Face
    st.subheader("🔑 Jeton Hugging Face")
    hf_token_found = HfFolder.get_token() or os.environ.get("HF_TOKEN")
    if hf_token_found:
        st.success("✅ Jeton HF trouvé et configuré.")
    else:
        st.warning("⚠️ Jeton HF non trouvé.")
    st.info("Il est recommandé de définir la variable d'environnement `HF_TOKEN` avec votre jeton personnel Hugging Face pour éviter les erreurs d'accès (403).")
    st.markdown("[Obtenir un jeton HF](https://huggingface.co/settings/tokens)")

    st.divider()

    # Gestion du modèle IA
    st.header("🤖 Modèle IA Gemma 3n")
    if st.session_state.model_loaded:
        st.success(f"{t('model_status')} {st.session_state.model_status}")
        if st.session_state.model and hasattr(st.session_state.model, 'device'):
            st.write(f"Device utilisé : `{st.session_state.model.device}`")
        
        col1_btn, col2_btn = st.columns(2)
        with col1_btn:
            if st.button("🔄 Recharger le modèle", type="secondary"):
                st.session_state.model = None
                st.session_state.processor = None
                st.session_state.model_loaded = False
                st.session_state.model_status = t("not_loaded")
                # Efface le cache de la fonction load_ai_model pour forcer le rechargement
                if 'load_ai_model' in st.cache_resource.__wrapped__.__wrapped__.__self__.__dict__:
                    st.cache_resource.__wrapped__.__wrapped__.__self__['load_ai_model'].clear()
                st.rerun()
        with col2_btn:
            # Bouton pour forcer la persistance via @st.cache_resource
            if st.button("💾 Forcer Persistance", type="secondary"):
                st.cache_resource.clear() # Efface le cache pour forcer le rechargement et la ré-application du cache
                st.success("Cache effacé. Le modèle sera rechargé et mis en cache la prochaine fois.")
                st.rerun()
    else:
        st.warning(f"{t('model_status')} {st.session_state.model_status}")
        if st.button(t("load_model_button"), type="primary"):
            try:
                model, processor = get_model_and_processor()
                st.session_state.model = model
                st.session_state.processor = processor
                st.session_state.model_loaded = True
                st.session_state.model_status = t("loaded")
                st.success("✅ Modèle chargé avec succès !")
            except Exception as e:
                st.session_state.model_status = f"{t('error')} : {e}"
                st.error(f"❌ Échec du chargement du modèle : {e}")
            st.rerun()

# --- ONGLET PRINCIPAUX ---
tab1, tab2, tab3, tab4 = st.tabs(t("tabs"))

# --- ONGLET 1: ANALYSE D'IMAGE ---
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
                st.error("Erreur : Le fichier est trop volumineux. Maximum 200MB.")
                uploaded_file = None
            elif uploaded_file.size == 0:
                st.error("Erreur : Le fichier est vide.")
                uploaded_file = None
            elif uploaded_file.size > (MAX_FILE_SIZE_BYTES * 0.8):
                st.warning("Attention : Le fichier est très volumineux, le chargement peut prendre du temps.")
    else:
        captured_image = st.camera_input("Prendre une photo de la plante", key="webcam_capture")
    
    image = None
    image_source = None
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_source = "upload"
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement de l'image uploadée : {e}")
    elif captured_image is not None:
        try:
            image = Image.open(captured_image)
            image_source = "webcam"
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement de l'image capturée : {e}")
    
    if image is not None:
        try:
            original_size = image.size
            image, was_resized = resize_image_if_needed(image, max_size=(800, 800))
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption=f"Image ({image_source})", use_container_width=True)
                if was_resized:
                    st.warning(f"⚠️ L'image a été redimensionnée de {original_size} à {image.size} pour optimiser le traitement.")
            
            with col2:
                st.markdown("**Informations de l'image :**")
                st.write(f"• Taille originale : {original_size[0]}x{original_size[1]} pixels")
                st.write(f"• Taille actuelle : {image.size[0]}x{image.size[1]} pixels")
                
            # --- AJOUT DU CHAMP POUR LES DÉTAILS DE LA PLANTE ET DES SYMPTÔMES ---
            user_details_prompt = st.text_area(
                "Décrivez la plante et les symptômes en détail (ex: 'feuilles de MANIOC avec taches jaunes...' ) :",
                placeholder="Ex: Feuilles de MANIOC avec taches jaunes sur les bords, texture poudreuse blanche au centre, nervures jaunies.",
                height=150
            )
            
            if st.button(t("analyze_button"), disabled=not st.session_state.model_loaded, type="primary"):
                if not st.session_state.model_loaded:
                    st.error("❌ Modèle non chargé. Veuillez le charger dans les réglages.")
                else:
                    with st.spinner("🔍 Analyse d'image en cours..."):
                        result = analyze_image_multilingual(image, user_details_prompt) # Passe les détails utilisateur
                    
                    st.markdown(t("analysis_results"))
                    st.markdown("---")
                    st.markdown(result)
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {e}")

# --- ONGLET 2: ANALYSE DE TEXTE ---
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

# --- ONGLET 3: MANUEL ---
with tab3:
    st.header(t("manual_title"))
    manual_content = {
        "fr": """
        ### 🚀 **Démarrage Rapide**
        1. **Charger le modèle** : Cliquez sur 'Charger le modèle IA' dans les réglages (sidebar). Le modèle Gemma 3n e2b it est gourmand en ressources. Si le chargement échoue, essayez une stratégie différente (ex: quantisation 8-bit ou CPU).
        2. **Choisir le mode** : Allez à l'onglet '📸 Analyse d'Image' ou '💬 Analyse de Texte'.
        3. **Soumettre votre demande** : Upload d'image, capture webcam, ou description textuelle détaillée. Pour l'image, décrivez les symptômes et le type de plante pour une meilleure précision.
        4. **Obtenir le diagnostic** : Lisez les résultats avec recommandations.
        
        ### 📸 **Analyse d'Image**
        *  **Formats acceptés** : PNG, JPG, JPEG.
        *  **Qualité** : Privilégiez des images claires, bien éclairées, avec le problème bien visible.
        *  **Redimensionnement** : Les images trop grandes sont automatiquement redimensionnées pour optimiser le traitement.
        *  **Détails** : Fournissez une description des symptômes et du type de plante dans le champ prévu à cet effet pour aider l'IA à mieux diagnostiquer.
        
        ### 💬 **Analyse de Texte**
        *  **Soyez précis** : Décrivez les symptômes, le type de plante (manioc, manguier, etc.), les conditions de culture, et les actions déjà tentées. Plus la description est détaillée et spécifique au MANIOC (si c'est le cas), plus le diagnostic sera pertinent.
        
        ### 🔍 **Interprétation des Résultats**
        *  Les résultats incluent un diagnostic potentiel, les causes probables, des recommandations de traitement et des conseils de prévention.
        *  Ces informations sont basées sur l'IA et doivent être considérées comme un guide. Consultez un expert pour des cas critiques.
        
        ### 💡 **Bonnes Pratiques**
        *  **Images multiples** : Si possible, prenez des photos sous différents angles.
        *  **Éclairage** : La lumière naturelle est idéale.
        *  **Focus** : Assurez-vous que la zone affectée est nette et bien visible.
        
        ### 💾 **Persistance du Modèle**
        *  L'application utilise `st.cache_resource` pour garder le modèle chargé en mémoire pendant la durée de votre session Streamlit, accélérant les analyses suivantes.
        *  Vous pouvez le recharger manuellement si nécessaire en utilisant le bouton "Forcer Persistance" (qui vide le cache) ou "Recharger le modèle".
        
        ### 🔒 **Jeton Hugging Face (HF_TOKEN)**
        *  Pour garantir la stabilité et la performance lors du téléchargement de modèles depuis Hugging Face, il est fortement recommandé de définir la variable d'environnement `HF_TOKEN`.
        *  Créez un jeton de lecture sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) et définissez-le dans votre environnement (ou utilisez `huggingface-cli login`) avant de lancer l'application.
        """,
        "en": """
        ### 🚀 **Quick Start**
        1. **Load the model** : Click 'Load AI Model' in the settings (sidebar). The Gemma 3n e2b it model is resource-intensive. If loading fails, try a different strategy (e.g., 8-bit quantization or CPU).
        2. **Choose mode** : Navigate to the '📸 Image Analysis' or '💬 Text Analysis' tab.
        3. **Submit your request** : Upload an image, capture via webcam, or provide a detailed text description. For images, describe symptoms and plant type for better accuracy.
        4. **Get diagnosis** : Read the results with recommendations.
        
        ### 📸 **Image Analysis**
        *  **Accepted formats** : PNG, JPG, JPEG.
        *  **Quality** : Prefer clear, well-lit images with the problem clearly visible.
        *  **Resizing** : Oversized images are automatically resized for processing optimization.
        *  **Details** : Provide a description of symptoms and plant type in the designated field to aid the AI's diagnosis.
        
        ### 💬 **Text Analysis**
        *  **Be specific** : Describe symptoms, plant type (manioc, mango, etc.), growing conditions, and actions already taken. More detail and specificity for MANIOC (if applicable) lead to better diagnosis.
        
        ### 🔍 **Result Interpretation**
        *  Results include a potential diagnosis, likely causes, treatment recommendations, and preventive advice.
        *  This AI-driven information is for guidance only. Consult a qualified expert for critical cases.
        
        ### 💡 **Best Practices**
        *  **Multiple images** : If possible, take photos from different angles.
        *  **Lighting** : Natural light is ideal.
        *  **Focus** : Ensure the affected area is sharp and clearly visible.
        
        ### 💾 **Model Persistence**
        *  The app uses `st.cache_resource` to keep the model loaded in memory during your Streamlit session, speeding up subsequent analyses.
        *  You can manually reload it if needed using the 'Force Persistence' (which clears the cache) or 'Reload Model' button.
        
        ### 🔒 **Hugging Face Token (HF_TOKEN)**
        *  To ensure stability and performance when downloading models from Hugging Face, it's highly recommended to set the environment variable `HF_TOKEN`.
        *  Create a read token on [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and set it in your environment (or use `huggingface-cli login`) before launching the application.
        """
    }
    st.markdown(manual_content[st.session_state.language])

# --- ONGLET 4: À PROPOS ---
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
    
    st.markdown("""
    • **Modèle** : Gemma 3n e2b it (Hugging Face - en ligne)
    • **Framework** : Streamlit
    • **Déploiement** : Hugging Face Spaces / En ligne
    """)
    
    st.markdown(f"### {t('creator_title')}")
    st.markdown(f"{t('creator_name')}")
    st.markdown(f"📍 {t('creator_location')}")
    st.markdown(f"📞 {t('creator_phone')}")
    st.markdown(f"📧 {t('creator_email')}")
    st.markdown(f"🔗 [{t('creator_linkedin')}](https://{t('creator_linkedin')})")
    
    st.markdown(f"### {t('competition_title')}")
    st.markdown(t("competition_text"))
    
    st.markdown("### ⚠️ Avertissement / Warning")
    st.markdown("Les résultats fournis par l'IA sont à titre indicatif uniquement et ne remplacent pas l'avis d'un expert agricole qualifié.")
    
    st.markdown("### 📞 Support")
    st.markdown("Pour toute question ou problème, consultez la documentation ou contactez le créateur.")

# --- PIED DE PAGE ---
st.markdown("---")
st.markdown(t("footer"))