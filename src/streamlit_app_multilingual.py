import streamlit as st
import os
import io
from PIL import Image
import requests
import torch
import google.generativeai as genai

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
        "load_model": "Charger le modèle Gemma 2B",
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
        "load_model": "Load Gemma 2B Model",
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

@st.cache_resource(show_spinner=False)
def load_model():
    """Charge le modèle Gemma 3n E4B IT depuis Hugging Face"""
    try:
        st.info("Chargement du modèle Gemma 3n E4B IT depuis Hugging Face...")
        
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        
        model_name = "google/gemma-3n-e4b-it"
        
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        st.success("Modèle Gemma 3n E4B IT chargé avec succès !")
        return model, processor
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None

def analyze_image_multilingual(image, prompt=""):
    """Analyse une image avec Gemma 3n E4B IT pour diagnostic précis"""
    try:
        # Vérifier si le modèle Gemma est chargé
        if not st.session_state.model_loaded:
            return "❌ Modèle Gemma non chargé. Veuillez d'abord charger le modèle dans les réglages."
        
        # Récupérer le modèle et le processeur
        model, processor = st.session_state.model, st.session_state.processor
        
        if not model or not processor:
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
    
    # Chargement du modèle
    if st.button(t("load_model"), type="primary"):
        with st.spinner("Chargement du modèle..." if st.session_state.language == "fr" else "Loading model..."):
            model, processor = load_model()
            if model and processor:
                st.session_state.model = model
                st.session_state.processor = processor
                st.session_state.model_loaded = True
                st.session_state.model_status = t("loaded")
                st.success("Modèle Gemma 3n E4B IT chargé avec succès !" if st.session_state.language == "fr" else "Gemma 3n E4B IT model loaded successfully!")
            else:
                st.session_state.model_loaded = False
                st.session_state.model_status = t("error")
                st.error("Échec du chargement du modèle" if st.session_state.language == "fr" else "Model loading failed")
    
    st.info(f"{t('model_status')} {st.session_state.model_status}")
    
    # Statut du modèle Gemma 3n E4B IT
    if st.session_state.model_loaded:
        st.success("✅ Modèle Gemma 3n E4B IT chargé")
        st.info("Le modèle est prêt pour l'analyse d'images et de texte")
    else:
        st.warning("⚠️ Modèle Gemma 3n E4B IT non chargé")
        st.info("Cliquez sur 'Charger le modèle' pour activer l'analyse")

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
    st.markdown("""
    • **Modèle** : Gemma 2B (version finale)
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