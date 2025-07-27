import streamlit as st
import os
import io
from PIL import Image
import requests
import torch

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - Diagnostic des Plantes",
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

@st.cache_resource(show_spinner=False)
def load_model():
    """Charge un modèle plus léger depuis Hugging Face"""
    try:
        st.info("Chargement du modèle Gemma 2B depuis Hugging Face...")
        
        # Import dynamique pour éviter les erreurs
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Utiliser un modèle plus léger (Gemma 2B au lieu de 3n)
        model_name = "google/gemma-2b-it"
        
        # Charger le tokenizer et le modèle
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

def analyze_image_light(image, prompt=""):
    """Analyse une image avec le modèle Gemma 2B (version légère)"""
    if not st.session_state.model_loaded:
        return "❌ Modèle non chargé. Veuillez le charger dans les réglages."
    
    try:
        model, tokenizer = st.session_state.model
        
        # Version légère : analyser seulement le texte
        if prompt:
            text_prompt = f"<start_of_turn>user\nAnalyse cette image de plante : {prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            text_prompt = "<start_of_turn>user\nAnalyse cette image de plante et décris les maladies présentes avec des recommandations pratiques.<end_of_turn>\n<start_of_turn>model\n"
        
        # Préparer l'entrée
        inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
        
        # Générer la réponse
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
        
        # Nettoyer la réponse
        response = response.replace("<end_of_turn>", "").strip()
        
        # Ajouter des recommandations si absentes
        if "recommandation" not in response.lower() and "action" not in response.lower():
            response += "\n\n**Recommandations ou actions urgentes :**\n• Isolez la plante malade si possible\n• Appliquez un traitement adapté\n• Surveillez les autres plantes\n• Consultez un expert si nécessaire"
        
        return response
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse d'image : {e}"

def analyze_text_light(text):
    """Analyse un texte avec le modèle Gemma 2B"""
    if not st.session_state.model_loaded:
        return "❌ Modèle non chargé. Veuillez le charger dans les réglages."
    
    try:
        model, tokenizer = st.session_state.model
        
        # Préparer l'entrée avec le format Gemma
        prompt = f"<start_of_turn>user\nTu es un assistant agricole expert. Analyse ce problème : {text}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Générer la réponse
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
st.title("🌱 AgriLens AI - Diagnostic des Plantes")
st.markdown("**Application de diagnostic des maladies de plantes avec IA**")

# Sidebar pour le chargement du modèle
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if st.button("Charger le modèle Gemma 2B (version légère)", type="primary"):
        with st.spinner("Chargement du modèle..."):
            model, tokenizer = load_model()
            if model and tokenizer:
                st.session_state.model = (model, tokenizer)
                st.session_state.model_loaded = True
                st.session_state.model_status = "✅ Chargé"
                st.success("Modèle chargé avec succès !")
            else:
                st.session_state.model_loaded = False
                st.session_state.model_status = "❌ Erreur"
                st.error("Échec du chargement du modèle")
    
    st.info(f"**Statut du modèle :** {st.session_state.model_status}")

# Onglets principaux
tab1, tab2, tab3 = st.tabs(["📸 Analyse d'Image", "💬 Analyse de Texte", "ℹ️ À propos"])

with tab1:
    st.header("🔍 Diagnostic par Image")
    st.markdown("Téléchargez une photo de plante malade pour obtenir un diagnostic")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Formats acceptés : PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Affichage de l'image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Image uploadée", use_container_width=True)
        
        with col2:
            st.markdown("**Informations de l'image :**")
            st.write(f"• Format : {image.format}")
            st.write(f"• Taille : {image.size[0]}x{image.size[1]} pixels")
            st.write(f"• Mode : {image.mode}")
        
        # Question spécifique
        question = st.text_area(
            "Question spécifique (optionnel) :",
            placeholder="Ex: Quelle est cette maladie ? Que faire pour la traiter ?",
            height=100
        )
        
        # Bouton d'analyse
        if st.button("🔬 Analyser avec l'IA", disabled=not st.session_state.model_loaded, type="primary"):
            if not st.session_state.model_loaded:
                st.error("❌ Veuillez d'abord charger le modèle dans les réglages")
            else:
                with st.spinner("🔍 Analyse en cours..."):
                    result = analyze_image_light(image, question)
                
                st.markdown("## 📊 Résultats de l'Analyse")
                st.markdown("---")
                st.markdown(result)

with tab2:
    st.header("💬 Diagnostic par Texte")
    st.markdown("Décrivez les symptômes de votre plante pour obtenir des conseils")
    
    text_input = st.text_area(
        "Description des symptômes :",
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
                result = analyze_text_light(text_input)
            
            st.markdown("## 📊 Résultats de l'Analyse")
            st.markdown("---")
            st.markdown(result)

with tab3:
    st.header("ℹ️ À propos d'AgriLens AI")
    st.markdown("""
    ### 🌱 Notre Mission
    AgriLens AI est une application de diagnostic des maladies de plantes utilisant l'intelligence artificielle 
    pour aider les agriculteurs à identifier et traiter les problèmes de leurs cultures.
    
    ### 🚀 Fonctionnalités
    - **Analyse d'images** : Diagnostic visuel des maladies
    - **Analyse de texte** : Conseils basés sur les descriptions
    - **Recommandations pratiques** : Actions concrètes à entreprendre
    - **Interface mobile** : Optimisée pour smartphones et tablettes
    
    ### 🔧 Technologie
    - **Modèle** : Gemma 2B (version légère)
    - **Framework** : Streamlit
    - **Déploiement** : Hugging Face Spaces
    
    ### ⚠️ Avertissement
    Les résultats fournis sont à titre indicatif uniquement. 
    Pour un diagnostic professionnel, consultez un expert qualifié.
    
    ### 📞 Support
    Pour toute question ou problème, consultez la documentation ou contactez l'équipe de développement.
    """)

# Footer
st.markdown("---")
st.markdown("*AgriLens AI - Diagnostic intelligent des plantes avec IA*") 