import streamlit as st
import os
import io
from PIL import Image
# import requests # Plus nécessaire si on n'utilise pas de requêtes externes pour d'autres IA
import torch
# import google.generativeai as genai # Supprimé car non utilisé par Gemma
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

# --- Traduction (Simplifiée pour cet exemple) ---
# Dans une vraie application, vous utiliseriez un système de traduction plus robuste.
# Pour cet exemple, nous allons utiliser un dictionnaire simple.
TRANSLATIONS = {
    "title": {"fr": "AgriLens AI", "en": "AgriLens AI"},
    "subtitle": {"fr": "Votre assistant IA pour le diagnostic des plantes", "en": "Your AI Assistant for Plant Diagnosis"},
    "config_title": {"fr": "Configuration", "en": "Configuration"},
    "load_model": {"fr": "Charger le modèle", "en": "Load Model"},
    "image_analysis_title": {"fr": "📸 Analyse d'Image", "en": "📸 Image Analysis"},
    "image_analysis_desc": {"fr": "Téléchargez une image de votre plante ou utilisez votre webcam pour obtenir un diagnostic.", "en": "Upload an image of your plant or use your webcam to get a diagnosis."},
    "choose_image": {"fr": "Choisissez une image", "en": "Choose an image"},
    "file_too_large_error": {"fr": "Erreur : Le fichier est trop volumineux. Maximum 200MB.", "en": "Error: File too large. Maximum 200MB."},
    "empty_file_error": {"fr": "Erreur : Le fichier est vide.", "en": "Error: File is empty."},
    "file_size_warning": {"fr": "Attention : Le fichier est très volumineux, le chargement peut prendre du temps.", "en": "Warning: File is very large, loading may take time."},
    "analyze_button": {"fr": "Analyser l'image", "en": "Analyze Image"},
    "analysis_results": {"fr": "Résultats de l'analyse :", "en": "Analysis Results:"},
    "text_analysis_title": {"fr": "💬 Analyse de Texte", "en": "💬 Text Analysis"},
    "text_analysis_desc": {"fr": "Décrivez les symptômes de votre plante pour obtenir des conseils.", "en": "Describe your plant's symptoms to get advice."},
    "symptoms_desc": {"fr": "Décrivez les symptômes ici...", "en": "Describe the symptoms here..."},
    "manual_title": {"fr": "📚 Manuel d'utilisation", "en": "📚 User Manual"},
    "about_title": {"fr": "ℹ️ À propos", "en": "ℹ️ About"},
    "footer": {"fr": "© 2024 AgriLens AI. Tous droits réservés.", "en": "© 2024 AgriLens AI. All rights reserved."}
}

def t(key):
    """Fonction de traduction simple."""
    lang = st.session_state.get('language', 'fr')
    return TRANSLATIONS.get(key, {}).get(lang, key) # Retourne la clé si la traduction n'existe pas

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
    except ImportError as e: issues.append(f"❌ Dépendance manquante : ")

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
        st.info(f"Chargement de  avec device_map='{device_map}', dtype=torch.{torch_dtype.__name__ if torch_dtype else 'None'}, quant='{quantization}'")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Configuration des arguments pour from_pretrained
        common_args = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "token": os.environ.get("HF_TOKEN") # Utilise le token si défini
        }
        
        # Ajout des arguments de quantification si spécifiés
        if quantization == "4bit":
            common_args.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16 # Souvent utilisé avec 4-bit
            })
        elif quantization == "8bit":
            common_args.update({"load_in_8bit": True})
        
        # Charger le processeur
        processor = AutoProcessor.from_pretrained(model_identifier, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
        
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
        raise ImportError(f"Bibliothèque manquante : . Installez-la avec `pip install transformers torch accelerate bitsandbytes`")
    except Exception as e:
        # Capturez l'exception spécifique ici pour un meilleur débogage
        raise Exception(f"Échec du chargement avec la stratégie  : ")

def load_model():
    """Charge le modèle Gemma 3n E4B IT (local ou Hugging Face) avec des stratégies robustes."""
    try:
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

        if is_local:
            # Stratégies pour le modèle local (priorité haute RAM)
            strategies_to_try.append(("Local (ultra-conservateur CPU)", lambda: load_model_strategy(LOCAL_MODEL_PATH, device_map="cpu", torch_dtype=torch.bfloat16, quantization=None, force_persistence=True)))
            strategies_to_try.append(("Local (conservateur CPU)", lambda: load_model_strategy(LOCAL_MODEL_PATH, device_map="cpu", torch_dtype=torch.bfloat16, quantization=None, force_persistence=True)))
        else:
            # Stratégies pour le modèle Hugging Face
            st.info("Modèle local non trouvé. Tentative de chargement depuis Hugging Face...")
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 # en GB
                st.info(f"Mémoire GPU disponible : {gpu_memory:.1f} GB")
                
                # Stratégies GPU basées sur la mémoire
                # Note: Gemma 3n E4B IT est assez grand, ajuster les seuils si nécessaire
                if gpu_memory >= 10: # 10 GB pour une expérience plus fluide avec float16
                    strategies_to_try.append(("Hugging Face (float16)", lambda: load_model_strategy(MODEL_ID_HF, device_map="auto", torch_dtype=torch.float16, quantization=None)))
                if gpu_memory >= 8: # 8 GB pour une version quantifiée 8-bit
                    strategies_to_try.append(("Hugging Face (8-bit quantization)", lambda: load_model_strategy(MODEL_ID_HF, device_map="auto", torch_dtype=torch.float16, quantization="8bit")))
                if gpu_memory >= 6: # 6 GB pour une version quantifiée 4-bit
                    strategies_to_try.append(("Hugging Face (4-bit quantization)", lambda: load_model_strategy(MODEL_ID_HF, device_map="auto", torch_dtype=torch.float16, quantization="4bit")))
                
                # Si la mémoire est moindre, une stratégie CPU est plus sûre
                if gpu_memory < 6:
                     st.warning("Mémoire GPU limitée. L'utilisation du CPU sera probablement plus stable.")

            # Stratégie CPU (par défaut ou si GPU insuffisant/absent)
            # Utiliser float32 pour plus de stabilité sur CPU
            strategies_to_try.append(("Hugging Face (conservative CPU)", lambda: load_model_strategy(MODEL_ID_HF, device_map="cpu", torch_dtype=torch.float32, quantization=None)))
            strategies_to_try.append(("Hugging Face (ultra-conservative CPU)", lambda: load_model_strategy(MODEL_ID_HF, device_map="cpu", torch_dtype=torch.float32, quantization=None)))

        # Boucle pour essayer chaque stratégie
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
                    # Ne pas continuer si c'est une erreur d'authentification critique
                    return None, None 
                else:
                    st.warning("Tentative suivante...")
                
                # Nettoyage mémoire après échec
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                continue # Passer à la stratégie suivante
        
        # Si toutes les stratégies échouent
        st.error("Toutes les stratégies de chargement du modèle ont échoué.")
        return None, None

    except ImportError as e:
        st.error(f"❌ Erreur de dépendance : . Installez avec `pip install transformers torch accelerate bitsandbytes`.")
        return None, None
    except Exception as e:
        st.error(f"❌ Une erreur générale s'est produite lors du chargement du modèle : ")
        return None, None

def analyze_image_multilingual(image, prompt=""):
    """Analyse une image avec Gemma 3n E4B IT pour un diagnostic précis."""
    if not st.session_state.model_loaded and not restore_model_from_cache():
        return "❌ Modèle Gemma non chargé. Veuillez d'abord charger le modèle dans les réglages."
    
    model, processor = st.session_state.model, st.session_state.processor
    if not model or not processor:
        return "❌ Modèle Gemma non disponible. Veuillez recharger le modèle."

    try:
        # Préparer le prompt textuel pour Gemma 3n
        # Le token <image> est crucial pour que le modèle sache où insérer les informations visuelles
        if st.session_state.language == "fr":
            gemma_prompt_text = f"<image>\nTu es un expert en pathologie végétale. Analyse cette image et réponds à la question : " if prompt else "<image>\nTu es un expert en pathologie végétale. Analyse cette image et fournis un diagnostic précis."
        else: # English
            gemma_prompt_text = f"<image>\nYou are an expert in plant pathology. Analyze this image and answer the question: " if prompt else "<image>\nYou are an expert in plant pathology. Analyze this image and provide a precise diagnosis."
        
        # Prétraiter l'image et le texte avec le processeur pour obtenir les inputs du modèle
        # C'est la méthode correcte pour les modèles multimodaux qui attendent des inputs combinés.
        processed_inputs = processor(
            images=[image], # L'image PIL doit être dans une liste
            text=gemma_prompt_text, # Le prompt textuel contenant <image>
            return_tensors="pt",
            padding=True
        )

        # Déplacer les inputs au bon device
        device = getattr(model, 'device', 'cpu')
        if hasattr(processed_inputs, 'to'):
            processed_inputs = processed_inputs.to(device)
        
        # Générer la réponse
        with torch.inference_mode():
            generation = model.generate(
                **processed_inputs, # Passer les inputs préparés par le processor
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            # Décode la réponse complète générée
            response_text = processor.decode(generation[0], skip_special_tokens=True)

        # Nettoyer la réponse pour enlever le prompt répété et les tokens spéciaux
        final_response = response_text.strip()
        
        # Supprimer le prompt texte initial si le modèle l'a répété
        if gemma_prompt_text.strip() in final_response:
            # Trouver la première occurrence du prompt textuel et prendre ce qui suit
            try:
                prompt_start_index = final_response.index(gemma_prompt_text.strip())
                final_response = final_response[prompt_start_index + len(gemma_prompt_text.strip()):].strip()
            except ValueError:
                # Si le prompt n'est pas trouvé exactement (ce qui est peu probable ici), on le laisse tel quel
                pass
            
        # Retirer les tokens spéciaux inutiles de la réponse
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
            
    except Exception as e:
        error_message = str(e)
        if "403" in error_message or "Forbidden" in error_message:
            return "❌ Erreur 403 - Accès refusé. Veuillez vérifier votre jeton Hugging Face (HF_TOKEN) et les quotas."
        elif "Number of images does not match number of special image tokens" in error_message:
            return "❌ Erreur : Le modèle n'a pas pu lier l'image au texte. Assurez-vous que le prompt contient bien le token `<image>` et que l'image est dans un format standard."
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
            prompt_template = f"Tu es un assistant agricole expert. Analyse ce problème de plante : \n\n**Instructions :**\n1. **Diagnostic** : Quel est le problème principal ?\n2. **Causes** : Quelles sont les causes possibles ?\n3. **Traitement** : Quelles sont les actions à entreprendre ?\n4. **Prévention** : Comment éviter le problème à l'avenir ?"
        else:
            prompt_template = f"You are an expert agricultural assistant. Analyze this plant problem: \n\n**Instructions:**\n1. **Diagnosis**: What is the main problem?\n2. **Causes**: What are the possible causes?\n3. **Treatment**: What actions should be taken?\n4. **Prevention**: How to avoid the problem in the future?"
        
        full_prompt = f"{prompt_template}\n\n**Description du problème :**\n{text}"

        # Préparer les messages pour le modèle Gemma (format conversationnel)
        messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}]
        
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
                max_new_tokens=500, # Augmenté pour des réponses plus complètes
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            # Décoder uniquement la partie générée (après le prompt)
            # Il est important de trouver le bon décalage pour ne décoder que la réponse.
            # `apply_chat_template` peut ajouter des tokens de début/fin de tour.
            # Une approche plus sûre est de décoder toute la séquence et de voir ce qu'on obtient.
            
            # Si apply_chat_template ajoute des tokens, on pourrait les retirer
            # Mais pour simplifier, décodons toute la séquence générée.
            response = processor.decode(generation[0], skip_special_tokens=True)

            # On essaie de retirer le prompt initial si le modèle l'a répété
            # Le format de chat de Gemma peut inclure <start_of_turn>, <end_of_turn>, etc.
            # Il faut être prudent lors du nettoyage.
            
            # La méthode la plus simple est de décoder et de nettoyer les tokens spéciaux
            # Si le prompt_template est répété, il faudra le retirer manuellement.
            cleaned_response = response.strip()
            # On peut tenter de retirer le prompt de base, mais attention aux variations
            if prompt_template in cleaned_response:
                cleaned_response = cleaned_response.split(prompt_template)[-1].strip()
            
            # Retirer les tokens de chat spécifiques si présents
            cleaned_response = cleaned_response.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()

        return cleaned_response
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse de texte : {e}"

# --- Interface Principale ---
st.title(t("title"))
st.markdown(t("subtitle"))

# --- Initialisation et Vérifications au démarrage ---
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
    # Assurer que st.session_state.language existe
    if 'language' not in st.session_state:
        st.session_state.language = 'fr'
    current_lang_index = 0 if st.session_state.language == "fr" else 1
    
    language = st.selectbox(
        "Sélectionnez votre langue :",
        language_options,
        index=current_lang_index,
        help="Change la langue de l'interface et des réponses de l'IA."
    )
    # Mettre à jour la langue dans st.session_state et recharger si changement
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
            st.write(f"Heure de chargement : ")
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

    # Affichage du statut de persistance
    st.divider()
    st.subheader("💾 Persistance du Modèle")
    if st.session_state.model_loaded and st.session_state.model_persistence_check:
        st.success("✅ Modèle chargé et persistant en cache.")
    elif st.session_state.model_loaded:
        st.warning("⚠️ Modèle chargé mais non persistant. Cliquez sur 'Forcer Persistance'.")
    else:
        st.warning("⚠️ Modèle non chargé.")


# --- Onglets Principaux ---
tab1, tab2, tab3, tab4 = st.tabs([t("image_analysis_title"), t("text_analysis_title"), t("manual_title"), t("about_title")])

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
    
    # Détecter l'environnement pour l'affichage
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
        • **Déploiement** : Hugging Face Spaces / en ligne
        """)
    
    # Informations du créateur
    st.markdown(f"### 👨‍💻 Créateur de l'Application / Application Creator")
    st.markdown(f"**Sidoine Kolaolé YEBADOKPO**")
    st.markdown(f"📍 Bohicon, République du Bénin")
    st.markdown(f"📧 syebadokpo@gmail.com")
    st.markdown(f"🔗 [linkedin.com/in/sidoineko](https://linkedin.com/in/sidoineko)")
    st.markdown(f"📁 [Hugging Face Portfolio](https://huggingface.co/Sidoineko)")
    
    # Informations de compétition
    st.markdown(f"### 🏆 Version Compétition Kaggle / Kaggle Competition Version")
    st.markdown("Cette première version d'AgriLens AI a été développée spécifiquement pour participer à une compétition Kaggle.")
    
    st.markdown("### ⚠️ Avertissement / Warning")
    st.markdown("Les résultats fournis par l'IA sont à titre indicatif uniquement et ne remplacent pas l'avis d'un expert agricole qualifié.")
    
    st.markdown("### 📞 Support")
    st.markdown("Pour toute question ou problème, consultez la documentation ou contactez le créateur.")

# --- Pied de page ---
st.markdown("---")
st.markdown("© 2024 AgriLens AI. Tous droits réservés.")