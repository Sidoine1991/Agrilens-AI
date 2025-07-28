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

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - Analyse de Plantes",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation des variables de session
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

def check_model_health():
    """Vérifie si le modèle est fonctionnel"""
    try:
        return (st.session_state.model is not None and 
                st.session_state.processor is not None and
                hasattr(st.session_state.model, 'device'))
    except Exception:
        return False

def diagnose_loading_issues():
    """Diagnostique les problèmes potentiels"""
    issues = []
    
    ram_gb = psutil.virtual_memory().total / (1024**3)
    if ram_gb < 8:
        issues.append(f"⚠️ RAM faible: {ram_gb:.1f}GB (recommandé: 8GB+)")
    
    disk_usage = psutil.disk_usage('/')
    disk_gb = disk_usage.free / (1024**3)
    if disk_gb < 10:
        issues.append(f"⚠️ Espace disque faible: {disk_gb:.1f}GB libre")
    
    try:
        requests.get("https://huggingface.co", timeout=5)
    except:
        issues.append("⚠️ Problème de connexion à Hugging Face")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 4:
            issues.append(f"⚠️ GPU mémoire faible: {gpu_memory:.1f}GB")
    else:
        issues.append("ℹ️ CUDA non disponible - CPU uniquement")
    
    return issues

def resize_image_if_needed(image, max_size=(800, 800)):
    """Redimensionne l'image si nécessaire"""
    original_size = image.size
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image, True
    return image, False

def afficher_ram_disponible(context=""):
    """Affiche l'utilisation de la RAM"""
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)
    ram_percent = ram.percent
    st.write(f"💾 RAM {context}: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram_percent:.1f}%)")

# Traductions
def t(key):
    translations = {
        "fr": {
            "title": "🌱 AgriLens AI - Assistant d'Analyse de Plantes",
            "subtitle": "Analysez vos plantes avec l'IA pour détecter les maladies",
            "tabs": ["📸 Analyse d'Image", "📝 Analyse de Texte", "⚙️ Configuration", "ℹ️ À Propos"],
            "image_analysis_title": "📸 Analyse d'Image de Plante",
            "image_analysis_desc": "Téléchargez ou capturez une image de votre plante",
            "choose_image": "Choisissez une image de plante...",
            "text_analysis_title": "📝 Analyse de Description Textuelle",
            "text_analysis_desc": "Décrivez les symptômes de votre plante",
            "enter_description": "Décrivez les symptômes de votre plante...",
            "config_title": "⚙️ Configuration",
            "about_title": "ℹ️ À Propos",
            "load_model": "Charger le Modèle",
            "model_status": "Statut du Modèle"
        },
        "en": {
            "title": "🌱 AgriLens AI - Plant Analysis Assistant",
            "subtitle": "Analyze your plants with AI to detect diseases",
            "tabs": ["📸 Image Analysis", "📝 Text Analysis", "⚙️ Configuration", "ℹ️ About"],
            "image_analysis_title": "📸 Plant Image Analysis",
            "image_analysis_desc": "Upload or capture an image of your plant",
            "choose_image": "Choose a plant image...",
            "text_analysis_title": "📝 Textual Description Analysis",
            "text_analysis_desc": "Describe your plant symptoms",
            "enter_description": "Describe your plant symptoms...",
            "config_title": "⚙️ Configuration",
            "about_title": "ℹ️ About",
            "load_model": "Load Model",
            "model_status": "Model Status"
        }
    }
    return translations[st.session_state.language].get(key, key)

def load_model():
    """Charge le modèle avec gestion d'erreurs améliorée et priorité au cache local"""
    try:
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        
        if st.session_state.load_attempt_count >= 3:
            st.error("🔄 Trop de tentatives de chargement. Redémarrez l'application.")
            return None, None
        st.session_state.load_attempt_count += 1
        st.info("🔍 Diagnostic de l'environnement...")
        issues = diagnose_loading_issues()
        if issues:
            with st.expander("📊 Diagnostic système", expanded=False):
                for issue in issues:
                    st.write(issue)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Utilisation d'un chemin local explicite pour le modèle
        model_id = "models/gemma-3n-E4B-it"
        if os.path.exists(model_id):
            st.info("Chargement du modèle depuis le dossier local explicite...")
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            model = Gemma3nForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
            st.success("✅ Modèle chargé avec succès (local explicite)")
        else:
            st.error(f"❌ Dossier du modèle local non trouvé : {model_id}")
            return None, None
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.model_loaded = True
        st.session_state.model_status = "Chargé"
        st.session_state.model_load_time = time.time()
        st.session_state.load_attempt_count = 0
        return model, processor
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None, None

def analyze_image_multilingual(image, prompt=""):
    """Analyse une image avec le modèle Gemma"""
    try:
        if not st.session_state.model_loaded or not check_model_health():
            st.error("❌ Modèle non chargé ou corrompu")
            return None
        
        # Préparer l'image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Préparer le prompt
        if not prompt:
            prompt = """Analysez cette image de plante et identifiez :
1. L'état de santé général de la plante
2. Les maladies ou problèmes visibles
3. Les recommandations de traitement
4. Les mesures préventives

Répondez en français de manière claire et structurée."""
        
        # Encoder l'image
        inputs = st.session_state.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        
        # Générer la réponse
        with st.spinner("🔍 Analyse en cours..."):
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Décoder la réponse
        response = st.session_state.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire seulement la partie générée
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        
        return response
        
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse: {e}")
        return None

def analyze_text_multilingual(text):
    """Analyse un texte descriptif avec le modèle Gemma"""
    try:
        if not st.session_state.model_loaded or not check_model_health():
            st.error("❌ Modèle non chargé ou corrompu")
            return None
        
        # Préparer le prompt
        prompt = f"""Analysez cette description de symptômes de plante et fournissez un diagnostic :

Description : {text}

Veuillez analyser et fournir :
1. Diagnostic probable
2. Causes possibles
3. Traitements recommandés
4. Mesures préventives

Répondez en français de manière claire et structurée."""
        
        # Encoder le texte
        inputs = st.session_state.processor(
            text=prompt,
            return_tensors="pt"
        )
        
        # Générer la réponse
        with st.spinner("🔍 Analyse en cours..."):
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Décoder la réponse
        response = st.session_state.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire seulement la partie générée
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        
        return response
        
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse: {e}")
        return None

# Interface principale
st.title(t("title"))
st.markdown(t("subtitle"))

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Sélection de langue
    language = st.selectbox(
        "🌐 Langue / Language",
        ["Français", "English"],
        index=0 if st.session_state.language == "fr" else 1
    )
    st.session_state.language = "fr" if language == "Français" else "en"
    
    st.divider()
    
    # Gestion du modèle
    st.header(t("model_status"))
    
    if st.session_state.model_loaded and check_model_health():
        st.success("✅ Modèle chargé et fonctionnel")
        st.write(f"**Statut :** {st.session_state.model_status}")
        if st.session_state.model_load_time:
            load_time_str = time.strftime('%H:%M:%S', time.localtime(st.session_state.model_load_time))
            st.write(f"**Heure de chargement :** {load_time_str}")
        
        # Bouton de rechargement (sans rerun automatique)
        if st.button("🔄 Recharger le modèle", type="secondary"):
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.session_state.processor = None
            st.session_state.load_attempt_count = 0
            st.info("🔄 Modèle déchargé. Cliquez sur 'Charger le modèle' pour recharger.")
    else:
        st.warning("⚠️ Modèle non chargé")
        
        # Bouton de chargement
        if st.button(t("load_model"), type="primary"):
            with st.spinner("🔄 Chargement du modèle..."):
                model, processor = load_model()
                if model is not None and processor is not None:
                    st.success("✅ Modèle chargé avec succès!")
                    st.rerun()
                else:
                    st.error("❌ Échec du chargement du modèle")

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
        uploaded_file = st.file_uploader(
            t("choose_image"), 
            type=['png', 'jpg', 'jpeg'],
            help="Formats acceptés : PNG, JPG, JPEG (max 200MB)" if st.session_state.language == "fr" else "Accepted formats: PNG, JPG, JPEG (max 200MB)"
        )
    else:
        st.markdown("**📷 Capture d'image par webcam**" if st.session_state.language == "fr" else "**📷 Webcam Image Capture**")
        st.info("💡 Positionnez votre plante malade devant la webcam et cliquez sur 'Prendre une photo'" if st.session_state.language == "fr" else "💡 Position your diseased plant in front of the webcam and click 'Take Photo'")
        
        captured_image = st.camera_input(
            "Prendre une photo de la plante" if st.session_state.language == "fr" else "Take a photo of the plant"
        )
    
    # Traitement de l'image
    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement de l'image : {e}")
    elif captured_image is not None:
        try:
            image = Image.open(captured_image)
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement de l'image : {e}")
    
    if image is not None:
        # Redimensionner l'image si nécessaire
        original_size = image.size
        image, was_resized = resize_image_if_needed(image, max_size=(800, 800))
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Image à analyser", use_container_width=True)
            if was_resized:
                st.info(f"ℹ️ Image redimensionnée de {original_size} à {image.size}")
        
        with col2:
            if st.session_state.model_loaded and check_model_health():
                # Options d'analyse
                analysis_type = st.selectbox(
                    "Type d'analyse :" if st.session_state.language == "fr" else "Analysis type:",
                    ["Analyse générale" if st.session_state.language == "fr" else "General analysis",
                     "Diagnostic maladie" if st.session_state.language == "fr" else "Disease diagnosis",
                     "Conseils de soins" if st.session_state.language == "fr" else "Care advice"]
                )
                
                custom_prompt = st.text_area(
                    "Prompt personnalisé (optionnel) :" if st.session_state.language == "fr" else "Custom prompt (optional):",
                    value="",
                    height=100
                )
                
                if st.button("🔍 Analyser l'image", type="primary"):
                    # Préparer le prompt selon le type d'analyse
                    if analysis_type == "Analyse générale" or analysis_type == "General analysis":
                        prompt = """Analysez cette image de plante et identifiez :
1. L'état de santé général de la plante
2. Les maladies ou problèmes visibles
3. Les recommandations de traitement
4. Les mesures préventives

Répondez en français de manière claire et structurée."""
                    elif analysis_type == "Diagnostic maladie" or analysis_type == "Disease diagnosis":
                        prompt = """Diagnostiquez cette plante en vous concentrant sur les maladies :

1. Identifiez les symptômes visibles
2. Déterminez la maladie probable
3. Expliquez les causes
4. Proposez un traitement spécifique

Répondez en français de manière structurée."""
                    else:  # Conseils de soins
                        prompt = """Analysez cette plante et donnez des conseils de soins :

1. État général de la plante
2. Besoins en eau et lumière
3. Conseils d'entretien
4. Améliorations recommandées

Répondez en français de manière structurée."""
                    
                    # Utiliser le prompt personnalisé si fourni
                    if custom_prompt.strip():
                        prompt = custom_prompt
                    
                    # Analyser l'image
                    result = analyze_image_multilingual(image, prompt)
                    
                    if result:
                        st.success("✅ Analyse terminée !")
                        st.markdown("### 📋 Résultats de l'analyse")
                        st.markdown(result)
                    else:
                        st.error("❌ Échec de l'analyse")
            else:
                st.warning("⚠️ Modèle non chargé. Chargez le modèle dans la sidebar pour analyser l'image.")

with tab2:
    st.header(t("text_analysis_title"))
    st.markdown(t("text_analysis_desc"))
    
    # Zone de texte pour la description
    text_input = st.text_area(
        t("enter_description"),
        height=200,
        placeholder="Exemple : Les feuilles de ma plante deviennent jaunes et tombent. Il y a des taches brunes sur les feuilles..." if st.session_state.language == "fr" else "Example: My plant leaves are turning yellow and falling. There are brown spots on the leaves..."
    )
    
    if st.button("🔍 Analyser la description", type="primary"):
        if text_input.strip():
            if st.session_state.model_loaded and check_model_health():
                result = analyze_text_multilingual(text_input)
                
                if result:
                    st.success("✅ Analyse terminée !")
                    st.markdown("### 📋 Résultats de l'analyse")
                    st.markdown(result)
                else:
                    st.error("❌ Échec de l'analyse")
            else:
                st.warning("⚠️ Modèle non chargé. Chargez le modèle dans la sidebar pour analyser le texte.")
        else:
            st.warning("⚠️ Veuillez entrer une description de la plante.")

with tab3:
    st.header(t("config_title"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Paramètres système")
        
        # Affichage des informations système
        ram = psutil.virtual_memory()
        st.write(f"**RAM totale :** {ram.total / (1024**3):.1f} GB")
        st.write(f"**RAM utilisée :** {ram.used / (1024**3):.1f} GB ({ram.percent:.1f}%)")
        
        disk = psutil.disk_usage('/')
        st.write(f"**Espace disque libre :** {disk.free / (1024**3):.1f} GB")
        
        if torch.cuda.is_available():
            st.write(f"**GPU :** {torch.cuda.get_device_name(0)}")
            st.write(f"**Mémoire GPU :** {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            st.write("**GPU :** Non disponible (CPU uniquement)")
    
    with col2:
        st.subheader("📊 Statistiques du modèle")
        
        if st.session_state.model_loaded and check_model_health():
            st.write("**Statut :** ✅ Chargé et fonctionnel")
            st.write(f"**Type :** {type(st.session_state.model).__name__}")
            if hasattr(st.session_state.model, 'device'):
                st.write(f"**Device :** {st.session_state.model.device}")
            if st.session_state.model_load_time:
                load_time_str = time.strftime('%H:%M:%S', time.localtime(st.session_state.model_load_time))
                st.write(f"**Heure de chargement :** {load_time_str}")
        else:
            st.write("**Statut :** ❌ Non chargé")
            st.write("**Type :** N/A")
            st.write("**Device :** N/A")

with tab4:
    st.header(t("about_title"))
    
    st.markdown("""
    ## 🌱 AgriLens AI
    
    **AgriLens AI** est un assistant intelligent pour l'analyse de plantes utilisant l'intelligence artificielle.
    
    ### 🚀 Fonctionnalités
    
    - **Analyse d'images** : Détection automatique des maladies de plantes
    - **Analyse textuelle** : Diagnostic basé sur les descriptions de symptômes
    - **Recommandations** : Conseils de traitement et prévention
    - **Interface multilingue** : Français et Anglais
    
    ### 🤖 Modèles utilisés
    
    - **Google Gemma 3n E4B IT** : Modèle de vision et langage
    - **Hugging Face Transformers** : Framework d'IA
    - **Streamlit** : Interface utilisateur
    
    ### 📝 Utilisation
    
    1. Chargez le modèle dans la sidebar
    2. Uploadez une image ou capturez avec la webcam
    3. Obtenez une analyse détaillée de votre plante
    4. Suivez les recommandations de traitement
    
    ### 🔧 Support technique
    
    - **Mode local** : Modèles téléchargés localement
    - **Mode en ligne** : Modèles depuis Hugging Face
    - **Gestion mémoire** : Optimisation automatique
    
    ---
    
    *Développé avec ❤️ pour les amoureux des plantes*
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    🌱 AgriLens AI - Assistant d'Analyse de Plantes | 
    <a href='#' target='_blank'>Documentation</a> | 
    <a href='#' target='_blank'>Support</a>
</div>
""", unsafe_allow_html=True) 