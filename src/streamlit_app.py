import os
import streamlit as st
from PIL import Image
import torch
from transformers import pipeline
import time
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - Diagnostic des Plantes",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main {
        max-width: 1000px;
        padding: 2rem;
    }
    .title {
        color: #2e8b57;
        text-align: center;
    }
    .upload-box {
        border: 2px dashed #2e8b57;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #2e8b57;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False, ttl=3600)  # Cache pour 1 heure
def load_model():
    """Charge le modèle avec gestion du cache et du timeout"""
    try:
        # Vérification du token
        if "HF_TOKEN" not in os.environ:
            st.error("Token d'accès Hugging Face manquant")
            return None
            
        # Chemin du cache
        cache_dir = Path("./model_cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Configuration du modèle
        model_name = "google/gemma-3n-e4b-it"
        
        # Afficher un message de chargement
        with st.spinner('Chargement initial du modèle Gemma 3n (peut prendre plusieurs minutes)...'):
            pipe = pipeline(
                "image-text-to-text",
                model=model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                model_kwargs={
                    "trust_remote_code": True,
                    "token": os.environ["HF_TOKEN"],
                    "cache_dir": str(cache_dir.absolute())
                }
            )
        return pipe
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

def display_upload_section():
    """Affiche la section de téléchargement d'image"""
    st.markdown("### 📤 Téléchargez une photo de plante")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choisissez une image...",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("#### Conseils :")
        st.markdown("• Photo nette et bien éclairée\n• Cadrez la partie malade\n• Évitez les reflets")
    
    return uploaded_file

def process_image(image, model):
    """Traite l'image avec le modèle"""
    try:
        # Préparation du prompt
        prompt = """Analyse cette image de plante et identifie les maladies potentielles.
        Fournis une réponse structurée avec :
        1. Le nom de la plante (si identifiable)
        2. Les maladies ou problèmes détectés
        3. Le niveau de confiance
        4. Des recommandations de traitement
        """
        
        # Barre de progression
        progress_text = "Analyse en cours..."
        progress_bar = st.progress(0, text=progress_text)
        
        # Simulation de progression
        for percent_complete in range(100):
            time.sleep(0.05)  # Simulation de traitement
            progress_bar.progress(percent_complete + 1, text=progress_text)
        
        # Appel au modèle
        response = model(image, prompt=prompt, max_new_tokens=500)
        
        # Nettoyage de la barre de progression
        progress_bar.empty()
        
        return response[0]['generated_text']
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {str(e)}")
        return None

def main():
    # En-tête
    st.title("🌱 AgriLens AI - Diagnostic des Plantes")
    st.markdown("### Analysez les maladies de vos plantes en un instant")
    
    # Section d'information
    with st.expander("ℹ️ Comment ça marche ?"):
        st.markdown("""
        1. **Téléchargez** une photo d'une plante malade
        2. Notre IA **analyse** l'image
        3. Recevez un **diagnostic** et des **conseils de traitement**
        """)
    
    # Chargement du modèle
    model = load_model()
    
    if model is None:
        st.error("""
        ❌ Impossible de charger le modèle. Vérifiez que :
        - Vous êtes connecté à Internet
        - Votre token d'API Hugging Face est valide
        - Vous avez accepté les conditions d'utilisation du modèle Gemma 3n
        """)
        return
    
    # Section de téléchargement
    uploaded_file = display_upload_section()
    
    if uploaded_file is not None:
        # Affichage de l'image
        image = Image.open(uploaded_file)
        st.image(image, caption='Votre image', use_column_width=True)
        
        # Bouton d'analyse
        if st.button("🔍 Analyser l'image", type="primary", use_container_width=True):
            with st.spinner('Analyse en cours...'):
                result = process_image(image, model)
                
                if result:
                    # Affichage des résultats
                    st.markdown("### 🔍 Résultats de l'analyse")
                    st.markdown("---")
                    st.markdown(result)
                    
                    # Section de feedback
                    st.markdown("---")
                    st.markdown("### 📝 Votre avis compte !")
                    col1, col2, col3 = st.columns(3)
                    with col2:
                        if st.button("👍 Le diagnostic est pertinent"):
                            st.success("Merci pour votre retour !")
                        if st.button("👎 Le diagnostic est inexact"):
                            st.warning("Merci pour votre retour. Nous allons améliorer notre modèle.")
    else:
        # Section d'exemple si aucune image n'est téléchargée
        st.markdown("---")
        st.markdown("### 📸 Exemple d'image attendue")
        st.image("https://via.placeholder.com/600x400?text=Photo+d%27une+plante+malade", 
                use_column_width=True)
        st.caption("Exemple : Feuilles de tomate avec des taches brunes")

    # Pied de page
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>ℹ️ Ce diagnostic est fourni à titre informatif uniquement.</p>
        <p>Pour un diagnostic professionnel, consultez un agronome qualifié.</p>
        <p>Version 1.0.0 | © 2025 AgriLens AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
