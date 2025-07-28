import streamlit as st
import os
import io
from PIL import Image
import requests
import torch
import google.generativeai as genai
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

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
    st.success("✅ Gemini API configurée")
else:
    gemini_model = None
    st.warning("⚠️ Gemini API non configurée - Ajoutez GOOGLE_API_KEY dans .env")

# Dictionnaires de traductions
translations = {
    "fr": {
        "title": "🌱 AgriLens AI - Diagnostic des Plantes",
        "subtitle": "**Application de diagnostic des maladies de plantes avec IA**",
        "config_title": "⚙️ Configuration",
        "load_model": "Charger le modèle Gemma 2B",
        "model_status": "**Statut du modèle :**",
        "not_loaded": "Non chargé",
        "loaded": "✅ Chargé",
        "error": "❌ Erreur",
        "tabs": ["📸 Analyse d'Image", "💬 Analyse de Texte", "📖 Manuel", "ℹ️ À propos"],
        "image_analysis_title": "🔍 Diagnostic par Image",
        "image_analysis_desc": "Téléchargez une photo de plante malade ou capturez-la avec votre webcam",
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
        "load_model": "Load Gemma 2B Model",
        "model_status": "**Model Status:**",
        "not_loaded": "Not loaded",
        "loaded": "✅ Loaded",
        "error": "❌ Error",
        "tabs": ["📸 Image Analysis", "💬 Text Analysis", "📖 Manual", "ℹ️ About"],
        "image_analysis_title": "🔍 Image Diagnosis",
        "image_analysis_desc": "Upload a photo of a diseased plant or capture it with your webcam",
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
        "creator_location": "Bohicon, Republic of Benin",
        "creator_phone": "+229 01 96 91 13 46",
        "creator_email": "syebadokpo@gmail.com",
        "creator_linkedin": "linkedin.com/in/sidoineko",
        "creator_portfolio": "Hugging Face Portfolio: Sidoineko/portfolio",
        "competition_title": "🏆 Kaggle Competition Version",
        "competition_text": "This first version of AgriLens AI was specifically developed to participate in the Kaggle competition. It represents our initial public production and demonstrates our expertise in AI applied to agriculture.",
        "footer": "*AgriLens AI - Intelligent plant disease diagnosis with AI*"
    }
}

def t(key):
    """Fonction helper pour récupérer les traductions"""
    return translations[st.session_state.language].get(key, key)

@st.cache_resource(show_spinner=False)
def load_model():
    """Charge le modèle Gemma 2B depuis Hugging Face"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "google/gemma-2b-it"
        
        # Charger le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Charger le modèle
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None, None

def analyze_image_multilingual(image, prompt=""):
    """Analyse une image avec Gemini pour diagnostic précis"""
    try:
        # Vérifier si Gemini est disponible
        if not gemini_model:
            return "❌ Gemini API non configurée. Veuillez configurer votre clé API Google pour analyser les images."
        
        # Préparer le prompt pour Gemini
        if st.session_state.language == "fr":
            if prompt:
                gemini_prompt = f"""
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
**Diagnostic Précis :** [Nom de la maladie et causes]

**Symptômes Détaillés :** [Liste des symptômes observés]

**Traitement Recommandé :** [Actions spécifiques à entreprendre]

**Actions Préventives :** [Mesures pour éviter la propagation]

**Niveau d'Urgence :** [Urgent/Modéré/Faible]

Réponds de manière structurée et précise.
"""
            else:
                gemini_prompt = """
Tu es un expert en pathologie végétale. Analyse cette image de plante et fournis un diagnostic précis.

**Instructions :**
1. **Diagnostic précis** : Identifie la maladie spécifique avec son nom scientifique
2. **Causes** : Explique les causes probables (champignons, bactéries, virus, carences, etc.)
3. **Symptômes détaillés** : Liste tous les symptômes observables dans l'image
4. **Traitement spécifique** : Donne des recommandations de traitement précises
5. **Actions préventives** : Conseils pour éviter la propagation
6. **Urgence** : Indique si c'est urgent ou non

**Format de réponse :**
**Diagnostic Précis :** [Nom de la maladie et causes]

**Symptômes Détaillés :** [Liste des symptômes observés]

**Traitement Recommandé :** [Actions spécifiques à entreprendre]

**Actions Préventives :** [Mesures pour éviter la propagation]

**Niveau d'Urgence :** [Urgent/Modéré/Faible]

Réponds de manière structurée et précise.
"""
        else:
            if prompt:
                gemini_prompt = f"""
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
**Precise Diagnosis:** [Disease name and causes]

**Detailed Symptoms:** [List of observed symptoms]

**Recommended Treatment:** [Specific actions to take]

**Preventive Actions:** [Measures to prevent spread]

**Urgency Level:** [Urgent/Moderate/Low]

Respond in a structured and precise manner.
"""
            else:
                gemini_prompt = """
You are an expert in plant pathology. Analyze this plant image and provide a precise diagnosis.

**Instructions:**
1. **Precise Diagnosis**: Identify the specific disease with its scientific name
2. **Causes**: Explain probable causes (fungi, bacteria, viruses, deficiencies, etc.)
3. **Detailed Symptoms**: List all observable symptoms in the image
4. **Specific Treatment**: Give precise treatment recommendations
5. **Preventive Actions**: Advice to prevent spread
6. **Urgency**: Indicate if urgent or not

**Response Format:**
**Precise Diagnosis:** [Disease name and causes]

**Detailed Symptoms:** [List of observed symptoms]

**Recommended Treatment:** [Specific actions to take]

**Preventive Actions:** [Measures to prevent spread]

**Urgency Level:** [Urgent/Moderate/Low]

Respond in a structured and precise manner.
"""
        
        # Analyser l'image directement avec Gemini
        response = gemini_model.generate_content([gemini_prompt, image])
        
        if st.session_state.language == "fr":
            return f"""
## 🧠 **Analyse par Gemini AI**
{response.text}
"""
        else:
            return f"""
## 🧠 **Analysis by Gemini AI**
{response.text}
"""
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse d'image : {e}"

def analyze_text_multilingual(text):
    """Analyse un texte avec le modèle Gemma 2B"""
    if not st.session_state.model_loaded:
        return "❌ Modèle non chargé. Veuillez le charger dans les réglages."
    
    try:
        model, tokenizer = st.session_state.model
        
        if st.session_state.language == "fr":
            prompt = f"<start_of_turn>user\nTu es un assistant agricole expert. Analyse ce problème : {text}<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = f"<start_of_turn>user\nYou are an expert agricultural assistant. Analyze this problem: {text}<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            generation = generation[0][inputs["input_ids"].shape[-1]:]
        
        response = tokenizer.decode(generation, skip_special_tokens=True)
        response = response.replace("<end_of_turn>", "").strip()
        return response
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse de texte : {e}"

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
    
    # Chargement du modèle
    if st.button(t("load_model"), type="primary"):
        with st.spinner("Chargement du modèle..." if st.session_state.language == "fr" else "Loading model..."):
            model, tokenizer = load_model()
            if model and tokenizer:
                st.session_state.model = (model, tokenizer)
                st.session_state.model_loaded = True
                st.session_state.model_status = t("loaded")
                st.success("Modèle chargé avec succès !" if st.session_state.language == "fr" else "Model loaded successfully!")
            else:
                st.session_state.model_loaded = False
                st.session_state.model_status = t("error")
                st.error("Échec du chargement du modèle" if st.session_state.language == "fr" else "Model loading failed")
    
    st.info(f"{t('model_status')} {st.session_state.model_status}")
    
    # Statut Gemini API
    if gemini_model:
        st.success("✅ Gemini API configurée")
    else:
        st.warning("⚠️ Gemini API non configurée")
        st.info("Ajoutez GOOGLE_API_KEY dans .env pour un diagnostic précis")

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
            st.markdown("**Informations de l'image :**" if st.session_state.language == "fr" else "**Image Information:**")
            st.write(f"• Format : {image.format}")
            st.write(f"• Taille originale : {original_size[0]}x{original_size[1]} pixels")
            st.write(f"• Taille actuelle : {image.size[0]}x{image.size[1]} pixels")
            st.write(f"• Mode : {image.mode}")
            
            if was_resized:
                st.warning("⚠️ L'image a été automatiquement redimensionnée pour optimiser le traitement")
        
        question = st.text_area(
            "Question spécifique (optionnel) :" if st.session_state.language == "fr" else "Specific question (optional):",
            placeholder="Ex: Quelle est cette maladie ? Que faire pour la traiter ?" if st.session_state.language == "fr" else "Ex: What is this disease? What should I do to treat it?",
            height=100
        )
        
        if st.button(t("analyze_button"), disabled=not gemini_model, type="primary"):
            if not gemini_model:
                st.error("❌ Gemini API non configurée. Veuillez configurer votre clé API Google pour analyser les images.")
                st.info("💡 L'analyse d'image nécessite Gemini API. Configurez GOOGLE_API_KEY dans .env")
            else:
                with st.spinner("🔍 Analyse en cours..."):
                    result = analyze_image_multilingual(image, question)
                
                st.markdown(t("analysis_results"))
                st.markdown("---")
                st.markdown(result)

with tab2:
    st.header(t("text_analysis_title"))
    st.markdown(t("text_analysis_desc"))
    
    text_input = st.text_area(
        t("symptoms_desc"),
        placeholder="Ex: Mes tomates ont des taches brunes sur les feuilles et les fruits..." if st.session_state.language == "fr" else "Ex: My tomatoes have brown spots on leaves and fruits...",
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
        
        1. **Choisissez votre langue** dans les réglages
        2. **Chargez le modèle AI** (bouton dans la sidebar)
        3. **Uploadez une image** ou **capturez avec la webcam**
        4. **Posez une question spécifique** (optionnel)
        5. **Obtenez votre diagnostic** avec recommandations
        
        ### 📸 **Analyse d'Image**
        
        **Deux méthodes disponibles :**
        - **📁 Upload** : Téléchargez une photo depuis votre appareil
        - **📷 Webcam** : Prenez une photo directement avec votre caméra
        
        **Conseils pour une meilleure analyse :**
        - Assurez-vous que la plante est bien éclairée
        - Photographiez les zones malades de près
        - Incluez plusieurs angles si possible
        
        ### 💬 **Analyse de Texte**
        
        Décrivez les symptômes observés :
        - Couleur des feuilles
        - Présence de taches ou déformations
        - État général de la plante
        - Conditions de culture
        
        ### 🎯 **Interprétation des Résultats**
        
        **Diagnostic Précis :** Nom scientifique de la maladie
        **Symptômes Détaillés :** Description complète des signes
        **Traitement Recommandé :** Actions spécifiques à entreprendre
        **Actions Préventives :** Mesures pour éviter la propagation
        **Niveau d'Urgence :** Priorité du traitement
        
        ### ⚠️ **Important**
        
        Cette application est un outil d'aide au diagnostic. Pour des cas critiques, consultez toujours un expert agricole local.
        """)
    else:
        st.markdown("""
        ### 🚀 **Quick Start**
        
        1. **Choose your language** in settings
        2. **Load the AI model** (button in sidebar)
        3. **Upload an image** or **capture with webcam**
        4. **Ask a specific question** (optional)
        5. **Get your diagnosis** with recommendations
        
        ### 📸 **Image Analysis**
        
        **Two methods available:**
        - **📁 Upload** : Upload a photo from your device
        - **📷 Webcam** : Take a photo directly with your camera
        
        **Tips for better analysis:**
        - Ensure the plant is well lit
        - Photograph diseased areas up close
        - Include multiple angles if possible
        
        ### 💬 **Text Analysis**
        
        Describe the observed symptoms:
        - Leaf color
        - Presence of spots or deformations
        - Overall plant condition
        - Growing conditions
        
        ### 🎯 **Interpreting Results**
        
        **Precise Diagnosis:** Scientific name of the disease
        **Detailed Symptoms:** Complete description of signs
        **Recommended Treatment:** Specific actions to take
        **Preventive Actions:** Measures to prevent spread
        **Urgency Level:** Treatment priority
        
        ### ⚠️ **Important**
        
        This application is a diagnostic aid tool. For critical cases, always consult a local agricultural expert.
        """)

with tab4:
    st.header(t("about_title"))
    
    if st.session_state.language == "fr":
        st.markdown(f"""
        ### {t('creator_title')}
        
        {t('creator_name')}
        
        **📍 Localisation :** {t('creator_location')}
        **📞 Téléphone :** {t('creator_phone')}
        **📧 Email :** {t('creator_email')}
        **💼 LinkedIn :** {t('creator_linkedin')}
        **🎯 Portfolio :** {t('creator_portfolio')}
        
        ### {t('competition_title')}
        
        {t('competition_text')}
        
        ### 🌟 **Fonctionnalités**
        
        - **IA Avancée** : Intégration Gemini pour diagnostic précis
        - **Capture Webcam** : Prise de photo directe sur le terrain
        - **Support Multilingue** : Français et Anglais
        - **Interface Mobile** : Optimisée pour smartphones
        - **Diagnostic Structuré** : Résultats organisés et clairs
        
        ### 🔬 **Technologies**
        
        - **Streamlit** : Interface web
        - **Gemini AI** : Analyse d'images avancée
        - **Gemma 2B** : Analyse de texte
        - **PyTorch** : Framework d'IA
        - **PIL** : Traitement d'images
        
        ### 📱 **Compatibilité**
        
        - **Ordinateurs** : Windows, Mac, Linux
        - **Smartphones** : Interface responsive
        - **Tablettes** : Optimisation tactile
        
        {t('footer')}
        """)
    else:
        st.markdown(f"""
        ### {t('creator_title')}
        
        {t('creator_name')}
        
        **📍 Location:** {t('creator_location')}
        **📞 Phone:** {t('creator_phone')}
        **📧 Email:** {t('creator_email')}
        **💼 LinkedIn:** {t('creator_linkedin')}
        **🎯 Portfolio:** {t('creator_portfolio')}
        
        ### {t('competition_title')}
        
        {t('competition_text')}
        
        ### 🌟 **Features**
        
        - **Advanced AI** : Gemini integration for precise diagnosis
        - **Webcam Capture** : Direct photo capture in the field
        - **Multilingual Support** : French and English
        - **Mobile Interface** : Optimized for smartphones
        - **Structured Diagnosis** : Organized and clear results
        
        ### 🔬 **Technologies**
        
        - **Streamlit** : Web interface
        - **Gemini AI** : Advanced image analysis
        - **Gemma 2B** : Text analysis
        - **PyTorch** : AI framework
        - **PIL** : Image processing
        
        ### 📱 **Compatibility**
        
        - **Computers** : Windows, Mac, Linux
        - **Smartphones** : Responsive interface
        - **Tablets** : Touch optimization
        
        {t('footer')}
        """)

# Footer
st.markdown("---")
st.markdown(t("footer")) 