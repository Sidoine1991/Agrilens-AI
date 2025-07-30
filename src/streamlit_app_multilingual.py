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
from transformers import AutoProcessor, Gemma3nForConditionalGeneration # Importations pour Hugging Face
from huggingface_hub import HfFolder # Pour vérifier le token HF

# --- Configuration de la Page ---
st.set_page_config(
    page_title="AgriLens AI - Diagnostic des Plantes",
    page_icon="🌱",
    layout="centered", # Centré pour une meilleure expérience sur toutes les tailles d'écran
    initial_sidebar_state="collapsed" # Sidebar fermée par défaut, peut être ouverte via le menu
)

# --- Cache global pour le modèle ---
# Ce cache permet au modèle de persister en mémoire entre les re-runs de Streamlit
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
            # Test simple pour vérifier que le modèle fonctionne
            if hasattr(st.session_state.model, 'device'):
                device = st.session_state.model.device # Accéder à l'attribut device
                return True
        return False
    except Exception:
        return False

def force_model_persistence():
    """Stocke le modèle et le processeur dans le cache global pour assurer la persistance."""
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
            if hasattr(cached_model, 'device'): # Vérifie si le modèle est toujours valide
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
    
    # Vérifier la présence du token Hugging Face
    if not HfFolder.get_token() and not os.environ.get("HF_TOKEN"):
        issues.append("⚠️ **Jeton Hugging Face (HF_TOKEN) non configuré.** Le téléchargement du modèle pourrait échouer ou être ralenti. Voir la section Configuration pour plus de détails.")

    # Vérifier les dépendances et les ressources système
    try:
        import transformers; issues.append(f"✅ Transformers v{transformers.__version__}")
        import torch; issues.append(f"✅ PyTorch v{torch.__version__}")
        if torch.cuda.is_available(): issues.append(f"✅ CUDA disponible : {torch.cuda.get_device_name(0)}")
        else: issues.append("⚠️ CUDA non disponible - utilisation CPU (plus lent)")
    except ImportError as e: issues.append(f"❌ Dépendance manquante : {e}")

    try:
        mem = psutil.virtual_memory()
        issues.append(f"💾 RAM disponible : {mem.available // (1024**3)} GB")
        if mem.available < 4 * 1024**3: # Moins de 4GB RAM est critique
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
        return resized_image, True # True: redimensionnée
    return image, False # False: non redimensionnée

def afficher_ram_disponible(context=""):
    """Affiche l'utilisation de la RAM."""
    try:
        mem = psutil.virtual_memory()
        st.info(f"💾 RAM {context}: {mem.available // (1024**3)} GB disponible")
        if mem.available < 4 * 1024**3:
            st.warning("⚠️ Moins de 4GB de RAM disponible, le chargement du modèle risque d'échouer !")
    except ImportError:
        st.warning("⚠️ Impossible de vérifier la RAM système.")

# --- Fonctions d'Analyse avec Gemma 3n E4B IT ---
MODEL_ID_HF = "google/gemma-3n-E4B-it"
LOCAL_MODEL_PATH = "D:/Dev/model_gemma" # Chemin vers votre modèle local (ajustez si nécessaire)

def load_model():
    """Charge le modèle Gemma 3n E4B IT (local ou Hugging Face) avec des stratégies robustes."""
    try:
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration

        # Diagnostic initial
        issues = diagnose_loading_issues()
        with st.expander("📊 Diagnostic système", expanded=False):
            for issue in issues:
                st.markdown(issue)

        # Nettoyer la mémoire avant le chargement
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Détecter si le modèle local existe
        is_local = os.path.exists(LOCAL_MODEL_PATH)

        # --- Stratégies de chargement ---
        strategies_to_try = []

        # 1. Modèle Local (si disponible)
        if is_local:
            strategies_to_try.append(("Local (ultra-conservateur CPU)", lambda: load_model_strategy(LOCAL_MODEL_PATH, device_map="cpu", torch_dtype=torch.bfloat16, quantization=None, force_persistence=True)))
            strategies_to_try.append(("Local (conservateur CPU)", lambda: load_model_strategy(LOCAL_MODEL_PATH, device_map="cpu", torch_dtype=torch.bfloat16, quantization=None, force_persistence=True))) # Ce sont les mêmes pour le local, mais pour cohérence
        else:
            # 2. Modèle Hugging Face (stratégies diverses)
            st.info("Modèle local non trouvé. Tentative de chargement depuis Hugging Face...")
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 # en GB
                st.info(f"Mémoire GPU disponible : {gpu_memory:.1f} GB")
                
                # Stratégies GPU basées sur la mémoire
                if gpu_memory >= 8:
                    strategies_to_try.append(("Hugging Face (custom memory)", lambda: load_model_strategy(MODEL_ID_HF, device_map="auto", torch_dtype=torch.float16, quantization=None, force_persistence=True)))
                if gpu_memory >= 4:
                    strategies_to_try.append(("Hugging Face (4-bit quantization)", lambda: load_model_strategy(MODEL_ID_HF, device_map="auto", torch_dtype=torch.float16, quantization="4bit", force_persistence=True)))
                if gpu_memory >= 6: # Les modèles Gemma sont plus à l'aise avec bfloat16 si supporté, sinon float16
                    strategies_to_try.append(("Hugging Face (8-bit quantization)", lambda: load_model_strategy(MODEL_ID_HF, device_map="auto", torch_dtype=torch.float16, quantization="8bit", force_persistence=True)))
                strategies_to_try.append(("Hugging Face (conservative GPU)", lambda: load_model_strategy(MODEL_ID_HF, device_map="auto", torch_dtype=torch.float16, quantization=None, force_persistence=True)))
            
            # Stratégie par défaut pour CPU si GPU absent ou trop petit
            strategies_to_try.append(("Hugging Face (conservative CPU)", lambda: load_model_strategy(MODEL_ID_HF, device_map="cpu", torch_dtype=torch.float32, quantization=None, force_persistence=True)))
            strategies_to_try.append(("Hugging Face (ultra-conservative CPU)", lambda: load_model_strategy(MODEL_ID_HF, device_map="cpu", torch_dtype=torch.float32, quantization=None, force_persistence=True))) # Double vérif CPU

        # Boucle pour essayer chaque stratégie
        for i, (name, strategy_func) in enumerate(strategies_to_try):
            st.info(f"Tentative {i+1}/{len(strategies_to_try)} : {name}...")
            try:
                model, processor = strategy_func()
                if model and processor:
                    st.success(f"✅ Modèle chargé avec succès via la stratégie : {name}")
                    return model, processor
            except Exception as e:
                error_msg = str(e)
                if "disk_offload" in error_msg.lower() or "out of memory" in error_msg.lower():
                    st.warning(f"La stratégie {name} a échoué (mémoire/disk_offload). Tentative suivante...")
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue
                elif "403" in error_msg or "Forbidden" in error_msg:
                    st.error(f"❌ Erreur d'accès Hugging Face (403) avec la stratégie {name}. Vérifiez votre HF_TOKEN.")
                    return None, None # Arrêter si c'est une erreur d'authentification
                else:
                    st.warning(f"La stratégie {name} a échoué : {error_msg}. Tentative suivante...")
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue
        
        # Si toutes les stratégies échouent
        st.error("Toutes les stratégies de chargement du modèle ont échoué.")
        return None, None

    except ImportError as e:
        st.error(f"❌ Erreur de dépendance : {e}. Assurez-vous que transformers et torch sont installés.")
        return None, None
    except Exception as e:
        st.error(f"❌ Une erreur générale s'est produite lors du chargement du modèle : {e}")
        return None, None

def load_model_strategy(model_identifier, device_map=None, torch_dtype=None, quantization=None, force_persistence=False):
    """
    Charge un modèle et son processeur en utilisant des paramètres spécifiques.
    Retourne le modèle et le processeur, ou (None, None) en cas d'échec.
    """
    try:
        st.info(f"Chargement de {model_identifier} avec device_map='{device_map}', dtype={torch_dtype}, quant={quantization}...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        afficher_ram_disponible("avant chargement")

        # Configuration des arguments pour from_pretrained
        common_args = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            "torch_dtype": torch_dtype,
        }
        
        # Ajout des arguments de quantification si spécifiés
        if quantization == "4bit":
            common_args.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16 # Souvent utilisé avec 4-bit
            })
        elif quantization == "8bit":
            common_args.update({"load_in_8bit": True})
        
        # Récupérer le token HF pour l'authentification
        hf_token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
        if hf_token:
            common_args["token"] = hf_token
            st.info("Utilisation du jeton Hugging Face pour l'authentification.")
        else:
            st.warning("Aucun jeton Hugging Face trouvé. L'accès aux modèles peut être limité.")

        # Charger le processeur
        processor = AutoProcessor.from_pretrained(model_identifier, trust_remote_code=True, token=hf_token)
        
        # Charger le modèle
        model = Gemma3nForConditionalGeneration.from_pretrained(model_identifier, **common_args)
        
        afficher_ram_disponible("après chargement")
        
        # Stocker dans session_state et forcer la persistance
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
        raise ImportError(f"Bibliothèque manquante : {e}. Installez-la avec `pip install transformers torch accelerate bitsandbytes`")
    except Exception as e:
        raise Exception(f"Échec du chargement avec la stratégie {model_identifier} : {e}")

def analyze_image_multilingual(image, prompt=""):
    """Analyse une image avec Gemma 3n E4B IT pour un diagnostic précis."""
    if not st.session_state.model_loaded and not restore_model_from_cache():
        return "❌ Modèle Gemma non chargé. Veuillez d'abord charger le modèle dans les réglages."
    
    model, processor = st.session_state.model, st.session_state.processor
    if not model or not processor:
        return "❌ Modèle Gemma non disponible. Veuillez recharger le modèle."

    try:
        # Définir le prompt textuel basé sur la langue et la question
        if st.session_state.language == "fr":
            gemma_prompt_text = f"Tu es un expert en pathologie végétale. Analyse cette image et réponds à la question : {prompt}" if prompt else "Tu es un expert en pathologie végétale. Analyse cette image et fournis un diagnostic précis."
        else:
            gemma_prompt_text = f"You are an expert in plant pathology. Analyze this image and answer the question: {prompt}" if prompt else "You are an expert in plant pathology. Analyze this image and provide a precise diagnosis."
        
        # Prétraiter l'image et le texte avec le processeur pour obtenir les inputs du modèle
        # Ceci est la méthode clé pour les modèles multimodaux comme Gemma3n
        processed_inputs = processor(
            images=[image], # L'image PIL doit être dans une liste
            text=gemma_prompt_text,
            return_tensors="pt",
            padding=True
        )

        # Déplacer les inputs au bon device
        device = getattr(model, 'device', 'cpu')
        if hasattr(processed_inputs, 'to'):
            processed_inputs = processed_inputs.to(device)
        
        # Générer la réponse en passant les inputs préparés
        # Les paramètres comme max_new_tokens, temperature, etc., sont gérés ici.
        with torch.inference_mode():
            generation = model.generate(
                **processed_inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            # Décode uniquement la partie générée par le modèle
            # Il faut trouver le décalage correct, souvent l'input_ids généré contient aussi le prompt.
            # L'erreur "0 image tokens" vient du fait que le modèle n'a pas bien intégré l'image.
            # En utilisant `processor(images=..., text=...)` on espère corriger ça.
            # Pour le décodage, on prend la réponse complète pour le moment.
            
            # Si la réponse contient le prompt, on pourrait essayer de le retirer.
            # C'est souvent le cas avec les modèles de chat.
            # Il faut décoder TOUS les tokens générés.
            response_text = processor.decode(generation[0], skip_special_tokens=True)

        # Nettoyer la réponse
        final_response = response_text.strip()
        
        # Supprimer le prompt texte si le modèle l'a répété au début de sa réponse
        if gemma_prompt_text.strip() in final_response:
            final_response = final_response.split(gemma_prompt_text.strip())[-1].strip()
            
        # Retirer les tokens spéciaux qui pourraient être visibles
        final_response = final_response.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()

        # Formater la sortie
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
            
    except ImportError:
        return "❌ Erreur : Les bibliothèques nécessaires (transformers, torch) ne sont pas installées."
    except Exception as e:
        error_message = str(e)
        if "403" in error_message or "Forbidden" in error_message:
            return "❌ Erreur 403 - Accès refusé. Veuillez vérifier votre jeton Hugging Face (HF_TOKEN) et les quotas."
        elif "Number of images does not match number of special image tokens" in error_message:
            return "❌ Erreur : Le modèle n'a pas pu traiter l'image. L'image n'est peut-être pas correctement intégrée au prompt ou le format n'est pas reconnu. Essayez avec une image plus simple ou un autre format."
        else:
            return f"❌ Erreur lors de l'analyse d'image : {e}"

def analyze_text_multilingual(text):
    """Analyse un texte avec le modèle Gemma 3n E4B IT."""
    if not st.session_state.model_loaded and not restore_model_from_cache():
        return "❌ Modèle non chargé. Veuillez le charger dans les réglages."
    
    model, processor = st.session_state.model, st.session_state.processor
    if not model or not processor:
        return "❌ Modèle Gemma non disponible. Veuillez recharger le modèle."
    
    try:
        # Définir le prompt basé sur la langue
        if st.session_state.language == "fr":
            prompt = f"Tu es un assistant agricole expert. Analyse ce problème de plante : {text}\n\n**Instructions :**\n1. **Diagnostic** : Quel est le problème principal ?\n2. **Causes** : Quelles sont les causes possibles ?\n3. **Traitement** : Quelles sont les actions à entreprendre ?\n4. **Prévention** : Comment éviter le problème à l'avenir ?"
        else:
            prompt = f"You are an expert agricultural assistant. Analyze this plant problem: {text}\n\n**Instructions:**\n1. **Diagnosis**: What is the main problem?\n2. **Causes**: What are the possible causes?\n3. **Treatment**: What actions should be taken?\n4. **Prevention**: How to avoid the problem in the future?"
        
        # Préparer les messages pour le modèle Gemma (format conversationnel)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        
        # Utiliser le processeur pour convertir le format conversationnel en tenseurs
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Déplacer les inputs au bon device
        device = getattr(model, 'device', 'cpu')
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        
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
            # Décoder uniquement la partie générée (après le prompt)
            input_len = inputs["input_ids"].shape[-1]
            generation = generation[0][input_len:]
            response = processor.decode(generation, skip_special_tokens=True)
        
        return response.strip()
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse de texte : {e}"

# --- Interface Principale ---
st.title(t("title"))
st.markdown(t("subtitle"))

# --- Initialisation et Vérifications ---
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_status' not in st.session_state:
    st.session_state.model_status = "Non chargé"

# Tenter de restaurer le modèle au chargement de l'application si non déjà chargé
if not st.session_state.model_loaded:
    if restore_model_from_cache():
        st.success("🔄 Modèle restauré automatiquement depuis le cache au démarrage.")
    else:
        st.info("💡 Cliquez sur 'Charger le modèle' dans les réglages pour commencer.")

# --- Sidebar pour la Configuration ---
with st.sidebar:
    st.header(t("config_title"))
    
    # Sélecteur de langue
    st.subheader("🌐 Langue / Language")
    language_options = ["Français", "English"]
    current_lang_index = 0 if st.session_state.language == "fr" else 1
    language = st.selectbox(
        "Sélectionnez votre langue :",
        language_options,
        index=current_lang_index,
        help="Change la langue de l'interface et des réponses de l'IA."
    )
    if st.session_state.language != ("fr" if language == "Français" else "en"):
        st.session_state.language = "fr" if language == "Français" else "en"
        st.rerun() # Recharge pour appliquer le changement de langue

    st.divider()

    # Statut du Jeton Hugging Face
    st.subheader("🔑 Jeton Hugging Face")
    hf_token_found = HfFolder.get_token() or os.environ.get("HF_TOKEN")
    if hf_token_found:
        st.success("✅ Jeton HF trouvé et configuré.")
    else:
        st.warning("⚠️ Jeton HF non trouvé.")
    st.info("Il est recommandé de définir la variable d'environnement `HF_TOKEN` avec votre jeton personnel Hugging Face pour éviter les erreurs d'accès (403).")
    st.markdown("[Obtenir un jeton HF](https://huggingface.co/settings/tokens)")

    st.divider()

    # Gestion du Modèle IA
    st.header("🤖 Modèle IA Gemma 3n")
    if st.session_state.model_loaded and check_model_persistence():
        st.success(f"✅ Modèle chargé ({st.session_state.model_status})")
        if st.session_state.model_load_time:
            load_time_str = time.strftime('%H:%M:%S', time.localtime(st.session_state.model_load_time))
            st.write(f"Heure de chargement : {load_time_str}")
        if hasattr(st.session_state.model, 'device'):
            st.write(f"Device utilisé : `{st.session_state.model.device}`")
        
        col1_btn, col2_btn = st.columns(2)
        with col1_btn:
            if st.button("🔄 Recharger le modèle", type="secondary"):
                st.session_state.model_loaded = False
                st.session_state.model = None
                st.session_state.processor = None
                st.session_state.global_model_cache.clear() # Vider le cache pour recharger
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
                st.rerun() # Recharger pour mettre à jour le statut

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
            MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024 # 200 MB
            if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                st.error(t("file_too_large_error"))
                uploaded_file = None
            elif uploaded_file.size == 0:
                st.error(t("empty_file_error"))
                uploaded_file = None
            elif uploaded_file.size > (MAX_FILE_SIZE_BYTES * 0.8): # Avertissement si très volumineux
                st.warning(t("file_size_warning"))
    else: # Webcam capture
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
            st.error(f"❌ Erreur lors du traitement de l'image uploadée : {e}")
            st.info("💡 Essayez avec une image différente ou un format différent (PNG, JPG, JPEG).")
    elif captured_image is not None:
        try:
            image = Image.open(captured_image)
            image_source = "webcam"
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement de l'image capturée : {e}")
            st.info("💡 Essayez de reprendre la photo.")
    
    if image is not None:
        try:
            original_size = image.size
            image, was_resized = resize_image_if_needed(image, max_size=(800, 800))
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption=f"Image ({image_source})" if image_source else "Image", use_container_width=True)
                if was_resized:
                    st.warning(f"⚠️ L'image a été redimensionnée de {original_size} à {image.size} pour optimiser le traitement.")
            
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
            st.error(f"Erreur lors du traitement de l'image : {e}")

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
    
    # Utilisation de la fonction de traduction pour le contenu du manuel
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
        2.  **Choose mode** : Go to '📸 Image Analysis' or '💬 Text Analysis' tab.
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
    • **Interface mobile** : Optimisée pour smartphones et tablettes
    • **Support multilingue** : Français et Anglais
    """)
    
    st.markdown("### 🔧 Technologie / Technology")
    
    # Détecter l'environnement pour l'affichage
    is_local = os.path.exists(LOCAL_MODEL_PATH)
    
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
        • **Déploiement** : Hugging Face Spaces / en ligne
        """)
    
    # Informations du créateur
    st.markdown(f"### 👨‍💻 Créateur de l'Application / Application Creator")
    st.markdown(f"**Sidoine Kolaolé YEBADOKPO**")
    st.markdown(f"📍 Bohicon, République du Bénin")
    st.markdown(f"📞 +229 01 96 91 13 46")
    st.markdown(f"📧 syebadokpo@gmail.com")
    st.markdown(f"🔗 [linkedin.com/in/sidoineko](https://linkedin.com/in/sidoineko)")
    st.markdown(f"📁 [Hugging Face Portfolio](https://huggingface.co/Sidoineko)")
    
    # Informations de compétition
    st.markdown(f"### 🏆 Version Compétition Kaggle / Kaggle Competition Version")
    st.markdown("Cette première version d'AgriLens AI a été développée spécifiquement pour participer à une compétition Kaggle.")
    
    st.markdown("### ⚠️ Avertissement / Warning")
    st.markdown("Les résultats fournis par l'IA sont à titre indicatif uniquement et ne remplacent pas l'avis d'un expert agricole qualifié.")

# --- Pied de page ---
st.markdown("---")
st.markdown(t("footer"))