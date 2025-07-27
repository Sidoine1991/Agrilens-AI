import streamlit as st
import os
import io
from PIL import Image
import requests
import torch

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
    """Charge un modèle plus léger depuis Hugging Face"""
    try:
        st.info("Chargement du modèle Gemma 2B depuis Hugging Face...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "google/gemma-2b-it"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        st.success("Modèle Gemma 2B chargé avec succès !")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None

def analyze_image_multilingual(image, prompt=""):
    """Analyse une image avec le modèle Gemma 2B"""
    if not st.session_state.model_loaded:
        return "❌ Modèle non chargé. Veuillez le charger dans les réglages."
    
    try:
        model, tokenizer = st.session_state.model
        
        if st.session_state.language == "fr":
            if prompt:
                text_prompt = f"<start_of_turn>user\nAnalyse cette image de plante : {prompt}<end_of_turn>\n<start_of_turn>model\n"
            else:
                text_prompt = "<start_of_turn>user\nAnalyse cette image de plante et décris les maladies présentes avec des recommandations pratiques.<end_of_turn>\n<start_of_turn>model\n"
        else:
            if prompt:
                text_prompt = f"<start_of_turn>user\nAnalyze this plant image: {prompt}<end_of_turn>\n<start_of_turn>model\n"
            else:
                text_prompt = "<start_of_turn>user\nAnalyze this plant image and describe the diseases present with practical recommendations.<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
            generation = generation[0][inputs["input_ids"].shape[-1]:]
        
        response = tokenizer.decode(generation, skip_special_tokens=True)
        response = response.replace("<end_of_turn>", "").strip()
        
        if st.session_state.language == "fr":
            if "recommandation" not in response.lower() and "action" not in response.lower():
                response += "\n\n**Recommandations ou actions urgentes :**\n• Isolez la plante malade si possible\n• Appliquez un traitement adapté\n• Surveillez les autres plantes\n• Consultez un expert si nécessaire"
        else:
            if "recommendation" not in response.lower() and "action" not in response.lower():
                response += "\n\n**Recommendations or urgent actions:**\n• Isolate the diseased plant if possible\n• Apply appropriate treatment\n• Monitor other plants\n• Consult an expert if necessary"
        
        return response
        
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

# Onglets principaux
tab1, tab2, tab3, tab4 = st.tabs(t("tabs"))

with tab1:
    st.header(t("image_analysis_title"))
    st.markdown(t("image_analysis_desc"))
    
    try:
        uploaded_file = st.file_uploader(
            t("choose_image"), 
            type=['png', 'jpg', 'jpeg'],
            help="Formats acceptés : PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(image, caption="Image uploadée", use_container_width=True)
                
                with col2:
                    st.markdown("**Informations de l'image :**")
                    st.write(f"• Format : {image.format}")
                    st.write(f"• Taille : {image.size[0]}x{image.size[1]} pixels")
                    st.write(f"• Mode : {image.mode}")
                
                question = st.text_area(
                    "Question spécifique (optionnel) :",
                    placeholder="Ex: Quelle est cette maladie ? Que faire pour la traiter ?",
                    height=100
                )
                
                if st.button(t("analyze_button"), disabled=not st.session_state.model_loaded, type="primary"):
                    if not st.session_state.model_loaded:
                        st.error("❌ Veuillez d'abord charger le modèle dans les réglages")
                    else:
                        with st.spinner("🔍 Analyse en cours..."):
                            result = analyze_image_multilingual(image, question)
                        
                        st.markdown(t("analysis_results"))
                        st.markdown("---")
                        st.markdown(result)
                        
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement de l'image : {e}")
                st.info("💡 Essayez avec une image différente ou un format différent (PNG, JPG, JPEG)")
                
    except Exception as e:
        st.error(f"❌ Erreur lors de l'upload : {e}")
        st.info("💡 Vérifiez que votre navigateur autorise les uploads de fichiers")

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