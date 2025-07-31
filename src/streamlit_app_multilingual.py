# --- IMPORTS ---
import streamlit as st
import os
import io
from PIL import Image
import requests
import torch
import gc
import time
import sys
import psutil
import traceback

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="AgriLens AI - Analyse de Plantes",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INITIALISATION DES VARIABLES DE SESSION ---
# Ces variables permettent de maintenir l'état de l'application entre les exécutions.
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'model_status' not in st.session_state:
    st.session_state.model_status = "Non chargé"
if 'model_load_time' not in st.session_state:
    st.session_state.model_load_time = None
if 'language' not in st.session_state:
    st.session_state.language = "fr"
if 'load_attempt_count' not in st.session_state:
    st.session_state.load_attempt_count = 0
if 'device' not in st.session_state:
    st.session_state.device = "cpu" # Valeur par défaut

# --- FONCTIONS D'AIDE SYSTÈME ---

def check_model_health():
    """Vérifie si le modèle et le processeur sont chargés et semblent opérationnels."""
    try:
        # S'assure que le modèle et le processeur existent et que le modèle a une propriété 'device'
        return (st.session_state.model is not None and
                st.session_state.processor is not None and
                hasattr(st.session_state.model, 'device'))
    except Exception:
        return False

def diagnose_loading_issues():
    """Diagnostique les problèmes potentiels avant le chargement du modèle (RAM, Disque, GPU)."""
    issues = []
    
    try:
        ram = psutil.virtual_memory()
        ram_gb = ram.total / (1024**3)
        if ram_gb < 8: # Seuil minimum recommandé
            issues.append(f"⚠️ RAM faible: {ram_gb:.1f}GB (recommandé: 8GB+)")
    except Exception:
        issues.append("⚠️ Impossible de vérifier la RAM.")
        
    try:
        disk_usage = psutil.disk_usage('/')
        disk_gb = disk_usage.free / (1024**3)
        if disk_gb < 10: # Seuil minimum recommandé
            issues.append(f"⚠️ Espace disque faible: {disk_gb:.1f}GB libre sur '/'")
    except Exception:
        issues.append("⚠️ Impossible de vérifier l'espace disque.")
        
    if torch.cuda.is_available():
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory < 6: # Seuil minimum recommandé pour des modèles comme Gemma
                issues.append(f"⚠️ GPU mémoire faible: {gpu_memory:.1f}GB (recommandé: 6GB+)")
        except Exception:
            issues.append("⚠️ Erreur lors de la vérification de la mémoire GPU.")
    else:
        issues.append("ℹ️ CUDA non disponible - Le modèle fonctionnera sur CPU (lentement)")
        
    return issues

def resize_image_if_needed(image, max_size=(1024, 1024)):
    """Redimensionne l'image si ses dimensions dépassent max_size pour optimiser l'analyse."""
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        # Utilise LANCZOS pour un redimensionnement de haute qualité
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image, True
    return image, False

def afficher_ram_disponible(context=""):
    """Affiche l'utilisation de la RAM de manière lisible dans l'interface Streamlit."""
    try:
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        ram_percent = ram.percent
        st.write(f"💾 RAM : {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram_percent:.1f}%)")
    except Exception:
        st.write("💾 Impossible d'afficher l'utilisation de la RAM.")

# --- GESTION DES TRADUCTIONS ---

def t(key):
    """Fonction utilitaire pour gérer les traductions de textes dans l'application."""
    translations = {
        "fr": {
            "title": "🌱 AgriLens AI - Assistant d'Analyse de Plantes",
            "subtitle": "Analysez vos plantes avec l'IA pour détecter les maladies",
            "tabs": ["📸 Analyse d'Image", "📝 Analyse de Texte", "⚙️ Configuration", "ℹ️ À Propos"],
            "image_analysis_title": "📸 Analyse d'Image de Plante",
            "image_analysis_desc": "Téléchargez ou capturez une image de votre plante pour obtenir un diagnostic.",
            "choose_image": "Choisissez une image de plante...",
            "text_analysis_title": "📝 Analyse de Description Textuelle",
            "text_analysis_desc": "Décrivez les symptômes de votre plante pour obtenir un diagnostic personnalisé.",
            "enter_description": "Décrivez les symptômes de votre plante ici...",
            "config_title": "⚙️ Configuration & Informations",
            "about_title": "ℹ️ À Propos de l'Application",
            "load_model": "Charger le Modèle IA",
            "model_status": "Statut du Modèle IA"
        },
        "en": {
            "title": "🌱 AgriLens AI - Plant Analysis Assistant",
            "subtitle": "Analyze your plants with AI to detect diseases",
            "tabs": ["📸 Image Analysis", "📝 Text Analysis", "⚙️ Configuration", "ℹ️ About"],
            "image_analysis_title": "📸 Plant Image Analysis",
            "image_analysis_desc": "Upload or capture an image of your plant for a diagnosis.",
            "choose_image": "Choose a plant image...",
            "text_analysis_title": "📝 Textual Description Analysis",
            "text_analysis_desc": "Describe your plant's symptoms for a personalized diagnosis.",
            "enter_description": "Describe your plant's symptoms here...",
            "config_title": "⚙️ Configuration & Information",
            "about_title": "ℹ️ About the Application",
            "load_model": "Load AI Model",
            "model_status": "AI Model Status"
        }
    }
    # Retourne le texte traduit ou la clé si la traduction n'existe pas
    return translations[st.session_state.language].get(key, key)

# --- CONSTANTES MODÈLES ---
# Modèle principal recommandé pour l'analyse de plantes, basé sur Gemma
MODEL_ID_HF = "google/gemma-3n-e2b-it"
# Modèle de fallback plus léger, utilisé si le modèle principal échoue (non implémenté dans ce code, mais présent pour l'exemple)
MODEL_ID_FALLBACK = "microsoft/DialoGPT-medium"

# --- FONCTIONS DE CHARGEMENT ET D'ANALYSE DU MODÈLE ---

def get_device_map():
    """Détermine si le modèle doit être chargé sur GPU ou CPU."""
    if torch.cuda.is_available():
        st.session_state.device = "cuda"
        # 'auto' permet à transformers de gérer la répartition sur les GPUs disponibles
        return "auto"
    else:
        st.session_state.device = "cpu"
        return "cpu"

def load_model():
    """
    Charge le modèle Gemma 3n et son processeur associé.
    Optimisé pour les environnements comme Hugging Face Spaces avec des ressources potentiellement limitées.
    """
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        # Limite le nombre de tentatives pour éviter les boucles infinies en cas d'échec persistant.
        if st.session_state.load_attempt_count >= 3:
            st.error("❌ Trop de tentatives de chargement ont échoué. Veuillez vérifier votre configuration et redémarrer l'application.")
            return None, None
        st.session_state.load_attempt_count += 1
        
        st.info("🔍 Diagnostic de l'environnement avant chargement...")
        issues = diagnose_loading_issues()
        if issues:
            with st.expander("📊 Diagnostic système", expanded=False):
                for issue in issues:
                    st.write(issue)
        
        # Nettoyage mémoire agressif pour libérer de la RAM et du cache GPU avant le chargement.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        processor = None
        model = None
        device_map = get_device_map() # Détermine si c'est CPU ou GPU
        
        try:
            st.info(f"Chargement du modèle depuis Hugging Face Hub : `{MODEL_ID_HF}`...")
            
            # Chargement du processeur (tokenizer, feature extractor, etc.)
            processor = AutoProcessor.from_pretrained(
                MODEL_ID_HF,
                trust_remote_code=True,
                cache_dir="/tmp/huggingface_cache" # Utilise un répertoire temporaire pour le cache
            )
            
            # Paramètres pour optimiser le chargement et l'inférence, surtout sur des ressources limitées.
            model_kwargs = {
                "torch_dtype": torch.float16, # Utilise float16 pour réduire l'empreinte mémoire GPU
                "trust_remote_code": True,
                "low_cpu_mem_usage": True, # Aide à réduire l'utilisation CPU lors du chargement
                "device_map": device_map, # "auto" pour répartir sur les GPUs, "cpu" sinon
                "cache_dir": "/tmp/huggingface_cache",
            }
            
            # Limite la mémoire GPU à 4GB si un GPU est disponible. Ceci est crucial pour les GPU avec moins de mémoire.
            if torch.cuda.is_available():
                model_kwargs["max_memory"] = {0: "4GB"}
            
            # Si on est sur CPU, on peut essayer d'autres optimisations, comme le offload sur disque.
            if device_map == "cpu":
                model_kwargs.update({
                    "torch_dtype": torch.float32, # float32 est souvent plus stable sur CPU
                    "offload_folder": "/tmp/model_offload" # Dossier pour décharger des parties du modèle si nécessaire
                })
            
            # Chargement du modèle lui-même
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID_HF,
                **model_kwargs
            )
            
            st.success(f"✅ Modèle `{MODEL_ID_HF}` chargé avec succès.")
            st.session_state.model_status = "Chargé (Hub)"
            st.session_state.model_loaded = True
            st.session_state.model_load_time = time.time()
            st.session_state.load_attempt_count = 0 # Réinitialise le compteur en cas de succès
            
            return model, processor
            
        except Exception as e:
            st.error(f"❌ Échec du chargement du modèle depuis Hugging Face Hub : ")
            st.error("💡 Conseil : Le modèle peut être trop volumineux pour les ressources disponibles (RAM/VRAM).")
            st.error(f"Détails de l'erreur : {e}")
            # Essayer de charger un modèle plus léger si le principal échoue (implémentation à ajouter si nécessaire)
            # if MODEL_ID_FALLBACK: ...
            return None, None
            
    except ImportError:
        st.error("❌ Erreur : Les bibliothèques `transformers` ou `torch` ne sont pas installées.")
        return None, None
    except Exception as e:
        st.error(f"❌ Erreur générale lors de l'initialisation du chargement du modèle : ")
        st.error(f"Détails de l'erreur : {e}")
        return None, None

# ==============================================================================
# FONCTION CORRIGÉE POUR L'ANALYSE D'IMAGE (MULTIMODALE)
# ==============================================================================

def analyze_image_multilingual(image, prompt_text=""):
    """
    Analyse une image de plante en utilisant le modèle Gemma et un prompt personnalisé.
    Utilise la méthode apply_chat_template pour une gestion robuste des entrées multimodales.
    """
    if not st.session_state.model_loaded or not check_model_health():
        st.error("❌ Modèle IA non chargé ou non fonctionnel. Veuillez le charger via la barre latérale.")
        return None
        
    try:
        # S'assurer que l'image est en mode RGB pour le traitement
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Préparation des messages au format attendu par apply_chat_template pour les modèles multimodaux
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image}, # L'image elle-même
                    {"type": "text", "text": prompt_text} # Le prompt texte associé
                ]
            }
        ]
        
        # Application du template de chat pour obtenir les inputs formatés pour le modèle
        inputs = st.session_state.processor.apply_chat_template(
            messages,
            add_generation_prompt=True, # Ajoute le préfixe pour la génération de réponse par le modèle
            tokenize=True,
            return_dict=True,
            return_tensors="pt" # Retourne des tensors PyTorch
        )
        
        # Déplacement des tensors vers le bon device (GPU ou CPU)
        inputs = {key: val.to(st.session_state.model.device) for key, val in inputs.items()}
        
        with st.spinner("🔍 Analyse d'image en cours... (peut prendre plusieurs minutes sur CPU)"):
            input_len = inputs["input_ids"].shape[-1] # Nombre de tokens en entrée
            
            # Génération de la réponse par le modèle
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=768, # Longueur maximale de la réponse générée
                do_sample=True, # Permet une génération plus créative
                temperature=0.7, # Contrôle le caractère aléatoire de la génération
                top_p=0.9 # Utilise le top-p sampling
            )
            
            # Extraction de la partie générée par le modèle (en excluant les tokens d'entrée)
            generation = outputs[0][input_len:]
            response = st.session_state.processor.decode(generation, skip_special_tokens=True) # Décodage en texte
            
            return response.strip() # Retourne le texte de réponse nettoyé
            
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse de l'image : ")
        st.error(traceback.format_exc()) # Affiche la trace complète de l'erreur pour le débogage
        return None

# ==============================================================================
# FONCTION CORRIGÉE POUR L'ANALYSE DE TEXTE
# ==============================================================================

def analyze_text_multilingual(text_description):
    """
    Analyse une description textuelle des symptômes d'une plante en utilisant le modèle Gemma.
    Retourne le diagnostic et les recommandations.
    """
    if not st.session_state.model_loaded or not check_model_health():
        st.error("❌ Modèle IA non chargé ou non fonctionnel. Veuillez le charger via la barre latérale.")
        return None
        
    try:
        # Construction d'un prompt structuré pour demander un diagnostic précis au modèle.
        prompt = f"""Analyse la description des symptômes de cette plante et fournis un diagnostic détaillé :
**Description des symptômes :**

**Instructions pour le diagnostic :**
1.  **Diagnostic probable :** Quel est le problème principal (maladie, carence, etc.) ?
2.  **Causes possibles :** Quelles sont les raisons derrière ces symptômes ?
3.  **Recommandations de traitement :** Quels traitements sont les plus adaptés ?
4.  **Conseils de soins préventifs :** Comment éviter que le problème ne se reproduise ?
Réponds de manière structurée, claire et en français. Ne répète pas la description des symptômes dans ta réponse."""
        
        # Préparation des entrées pour le modèle (ici, uniquement du texte).
        # Note: Pour les modèles chat comme Gemma, il est préférable d'utiliser apply_chat_template même pour le texte seul.
        # Ici, j'utilise directement le processor pour un exemple simple, mais apply_chat_template est plus robuste.
        # Si tu utilises `apply_chat_template` pour l'image, il faut aussi l'utiliser ici pour la cohérence.
        # Dans le code original, `analyze_image_multilingual` utilise `apply_chat_template`.
        # Pour simplifier et pour cet exemple, je vais adapter le prompt pour être utilisé avec `apply_chat_template`
        # comme le modèle est un modèle de chat.

        messages = [
            {"role": "user", "content": prompt}
        ]
        inputs = st.session_state.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Déplacement des tensors sur le bon device
        inputs = {key: val.to(st.session_state.model.device) for key, val in inputs.items()}
        
        # Générer la réponse
        with st.spinner("🔍 Analyse textuelle en cours..."):
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=512, # Longueur maximale de la réponse
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            # Décoder la réponse
            response = st.session_state.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extraire uniquement la partie générée par le modèle (en retirant le prompt initial)
            # Il faut trouver un moyen fiable de séparer le prompt de la réponse.
            # Si apply_chat_template formate bien, la réponse commence après les tokens du prompt.
            # Une méthode plus sûre serait de reconstruire le prompt formaté et de le chercher dans la réponse.
            # Pour cet exemple, je vais extraire la partie après le prompt original.
            
            # Trouve la position de fin du contenu du dernier message utilisateur
            # Cela suppose que `apply_chat_template` ajoute des séparateurs clairs.
            # Une approche plus robuste serait de séparer en fonction des tokens spéciaux ou du rôle.
            
            # Si le prompt est exactement le contenu du dernier message et que le modèle répond après:
            # Il faut faire attention si le modèle répète une partie du prompt.
            # On va se baser sur la longueur des tokens d'entrée pour extraire la partie générée.
            
            input_len = inputs["input_ids"].shape[-1]
            response_only_tokens = outputs[0][input_len:]
            response_only = st.session_state.processor.decode(response_only_tokens, skip_special_tokens=True)
            
            return response_only.strip()
            
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse textuelle : ")
        st.error(traceback.format_exc())
        return None

# --- INTERFACE UTILISATEUR STREAMLIT ---

st.title(t("title"))
st.markdown(t("subtitle"))

# --- BARRE LATÉRALE (SIDEBAR) POUR LA CONFIGURATION ---
with st.sidebar:
    st.header(t("config_title"))
    
    # Sélecteur de langue
    lang_selector_options = ["Français", "English"]
    current_lang_index = 0 if st.session_state.language == "fr" else 1
    language_selected = st.selectbox(
        "🌐 Langue / Language",
        lang_selector_options,
        index=current_lang_index,
        help="Sélectionnez la langue de l'interface et des réponses."
    )
    # Met à jour la langue dans la session_state si elle change
    st.session_state.language = "fr" if language_selected == "Français" else "en"
    
    st.divider()
    
    # Affichage du statut du modèle IA et bouton de chargement/rechargement
    st.header(t("model_status"))
    
    if st.session_state.model_loaded and check_model_health():
        st.success("✅ Modèle chargé et fonctionnel")
        st.write(f"**Statut :** `{st.session_state.model_status}`")
        if st.session_state.model_load_time:
            # Affiche l'heure de chargement de manière lisible
            load_time_str = time.strftime('%H:%M:%S', time.localtime(st.session_state.model_load_time))
            st.write(f"**Heure de chargement :** {load_time_str}")
        
        # Bouton pour recharger le modèle si nécessaire
        if st.button("🔄 Recharger le modèle", type="secondary"):
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.session_state.processor = None
            st.session_state.model_status = "Non chargé"
            st.session_state.load_attempt_count = 0
            st.info("Modèle déchargé. Cliquez sur 'Charger le modèle IA' pour le recharger.")
            st.rerun() # Relance l'application pour réinitialiser l'état
    else:
        st.warning("⚠️ Modèle IA non chargé")
        # Bouton pour charger le modèle
        if st.button(t("load_model"), type="primary"):
            with st.spinner("🔄 Chargement du modèle IA en cours..."):
                load_model() # Appelle la fonction de chargement
            st.rerun() # Relance l'application pour mettre à jour le statut

    st.divider()
    
    # Section pour afficher les ressources système
    st.subheader("📊 Ressources Système")
    afficher_ram_disponible() # Affiche l'utilisation de la RAM
    
    if torch.cuda.is_available():
        try:
            # Affiche l'utilisation de la mémoire GPU
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            st.write(f"🚀 GPU utilisation : {gpu_memory_allocated:.1f}GB / {gpu_total_memory:.1f}GB")
        except Exception:
            st.write("🚀 GPU : Non disponible (utilisation CPU)")
    else:
        st.write("🚀 GPU : Non disponible (utilisation CPU)")

# --- ONGLET PRINCIPAUX ---
# Crée les onglets pour organiser l'application
tab1, tab2, tab3, tab4 = st.tabs(t("tabs"))

# --- ONGLET 1: ANALYSE D'IMAGE ---
with tab1:
    st.header(t("image_analysis_title"))
    st.markdown(t("image_analysis_desc"))
    
    # Option pour choisir entre l'upload d'un fichier ou la capture via webcam
    capture_option = st.radio(
        "Choisissez votre méthode de capture :",
        ["📁 Upload d'image" if st.session_state.language == "fr" else "📁 Upload Image",
         "📷 Capture par webcam" if st.session_state.language == "fr" else "📷 Webcam Capture"],
        horizontal=True
    )
    
    image_to_analyze_data = None
    if capture_option.startswith("📁"):
        # Widget pour uploader un fichier image
        image_to_analyze_data = st.file_uploader(
            t("choose_image"),
            type=['png', 'jpg', 'jpeg']
        )
    else:
        # Widget pour capturer une image via la webcam
        image_to_analyze_data = st.camera_input("Prendre une photo de la plante")
    
    # Si une image a été fournie (uploadée ou capturée)
    if image_to_analyze_data:
        image_to_analyze = Image.open(image_to_analyze_data)
        original_size = image_to_analyze.size
        # Redimensionne l'image si nécessaire avant l'analyse
        resized_image, was_resized = resize_image_if_needed(image_to_analyze)
        
        # Utilise des colonnes pour afficher l'image et les options d'analyse côte à côte
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(resized_image, caption="Image à analyser", use_container_width=True)
            if was_resized:
                st.info(f"ℹ️ Image redimensionnée de {original_size} à {resized_image.size} pour l'analyse.")
        
        with col2:
            # Affiche les options d'analyse uniquement si le modèle est chargé
            if st.session_state.model_loaded and check_model_health():
                st.subheader("Options d'analyse")
                # Prompt par défaut pour une analyse détaillée de l'image
                default_prompt = """Analyse cette image de plante et fournis un diagnostic complet :
1.  **État général de la plante :** Décris son apparence globale.
2.  **Identification des problèmes :** Liste les maladies, parasites ou carences visibles.
3.  **Diagnostic probable :** Indique le problème le plus probable.
4.  **Causes possibles :** Explique ce qui a pu causer ce problème.
5.  **Recommandations de traitement :** Propose des solutions concrètes.
6.  **Conseils préventifs :** Donne des astuces pour éviter que le problème ne revienne."""
                
                # Champ pour un prompt personnalisé
                custom_prompt_input = st.text_area(
                    "Prompt personnalisé (optionnel) :",
                    value=default_prompt,
                    height=250,
                    placeholder="Entrez une requête spécifique ici..."
                )
                
                # Bouton pour lancer l'analyse de l'image
                if st.button("🔍 Analyser l'image", type="primary"):
                    analysis_result = analyze_image_multilingual(resized_image, prompt_text=custom_prompt_input)
                    
                    if analysis_result:
                        st.success("✅ Analyse terminée !")
                        st.markdown("### 📋 Résultats de l'analyse")
                        st.markdown(analysis_result) # Affiche le résultat de l'analyse
                    else:
                        st.error("❌ Échec de l'analyse de l'image.")
            else:
                st.warning("⚠️ Modèle IA non chargé. Veuillez d'abord charger le modèle depuis la barre latérale.")

# --- ONGLET 2: ANALYSE DE TEXTE ---
with tab2:
    st.header(t("text_analysis_title"))
    st.markdown(t("text_analysis_desc"))
    
    # Zone de texte pour saisir la description des symptômes
    text_description_input = st.text_area(
        t("enter_description"),
        height=200,
        placeholder="Ex: Feuilles de tomate avec des taches jaunes et brunes, les bords s'enroulent vers le haut..."
    )
    
    # Bouton pour lancer l'analyse textuelle
    if st.button("🔍 Analyser la description", type="primary"):
        if text_description_input.strip(): # Vérifie que le champ n'est pas vide
            if st.session_state.model_loaded and check_model_health():
                analysis_result = analyze_text_multilingual(text_description_input)
                
                if analysis_result:
                    st.success("✅ Analyse terminée !")
                    st.markdown("### 📋 Résultats de l'analyse")
                    st.markdown(analysis_result) # Affiche le résultat de l'analyse
                else:
                    st.error("❌ Échec de l'analyse textuelle.")
            else:
                st.warning("⚠️ Modèle IA non chargé. Veuillez d'abord charger le modèle depuis la barre latérale.")
        else:
            st.warning("⚠️ Veuillez entrer une description des symptômes.")

# --- ONGLET 3: CONFIGURATION & INFORMATIONS ---
with tab3:
    st.header(t("config_title"))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔧 Informations Système")
        try:
            # Affiche les informations système détaillées
            ram = psutil.virtual_memory()
            st.write(f"**RAM Totale :** {ram.total / (1024**3):.1f} GB")
            st.write(f"**RAM Utilisée :** {ram.used / (1024**3):.1f} GB ({ram.percent:.1f}%)")
            disk = psutil.disk_usage('/')
            st.write(f"**Espace Disque Libre (/) :** {disk.free / (1024**3):.1f} GB")
            if torch.cuda.is_available():
                st.write(f"**GPU Détecté :** {torch.cuda.get_device_name(0)}")
            else:
                st.write("**GPU :** Non disponible")
        except Exception as e:
            st.error(f"Erreur lors de la récupération des informations système : {e}")
            
    with col2:
        st.subheader("📊 Statistiques du Modèle IA")
        if st.session_state.model_loaded and check_model_health():
            st.write("**Statut :** ✅ Chargé et fonctionnel")
            st.write(f"**Type de modèle :** `{type(st.session_state.model).__name__}`")
            st.write(f"**Device utilisé :** `{st.session_state.model.device}`")
        else:
            st.write("**Statut :** ❌ Non chargé")

# --- ONGLET 4: À PROPOS ---
with tab4:
    st.header(t("about_title"))
    st.markdown("""
    ### 🌱 AgriLens AI : L'expert agronome dans votre poche
    **AgriLens AI** a été conçu pour le **Google - The Gemma 3n Impact Challenge**. Notre mission est de fournir aux agriculteurs et jardiniers du monde entier un outil de diagnostic puissant, gratuit et **fonctionnant sans connexion Internet**.
    #### Notre Vision
    Dans de nombreuses régions du monde, l'accès à l'expertise agricole est limité et la connectivité internet est peu fiable. Ces obstacles entraînent des pertes de récoltes qui auraient pu être évitées. AgriLens AI répond directement à ce problème en exploitant les capacités **offline et multimodales** du modèle Gemma 3n.
    #### Fonctionnalités Clés pour l'Impact
    -   **✅ 100% Offline :** Après le téléchargement initial du modèle, l'application fonctionne sans aucune connexion, garantissant l'accès et la confidentialité des données, même dans les zones les plus reculées.
    -   **📸 Analyse Visuelle Instantanée :** Prenez une photo de votre plante et obtenez un diagnostic détaillé en quelques instants.
    -   **🗣️ Support Multilingue :** L'interface et les prompts sont conçus pour être facilement traduisibles, brisant les barrières linguistiques.
    #### Technologie
    -   **Modèle IA :** Google Gemma 3n (`google/gemma-3n-e2b-it`).
    -   **Framework :** Streamlit pour une interface rapide et interactive.
    -   **Bibliothèques :** `transformers`, `torch`, `Pillow`, `psutil`.
    
    ---
    *Développé avec passion pour un impact durable dans le secteur agricole.*
    """)

# --- PIED DE PAGE ---
st.divider()
st.markdown("<div style='text-align: center; color: #666;'>Projet pour le Google - The Gemma 3n Impact Challenge</div>", unsafe_allow_html=True)