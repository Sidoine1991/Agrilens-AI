import os
import streamlit as st
from PIL import Image
import torch
from transformers import pipeline
import time
from pathlib import Path
import json
import logging
from fastapi import FastAPI, Request, Response
import uvicorn

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

# Variable globale pour le modèle
MODEL = None

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vérification de l'environnement
def check_environment():
    """Vérifie les variables d'environnement requises"""
    required_vars = ["HF_TOKEN"]
    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        logger.error(f"Variables d'environnement manquantes : {', '.join(missing_vars)}")
        return False
    return True

@st.cache_resource(show_spinner=False, ttl=3600)
def load_model():
    """Charge le modèle avec gestion du cache et du timeout"""
    global MODEL
    
    if MODEL is not None:
        return MODEL
        
    if not check_environment():
        st.error("Configuration manquante. Vérifiez les logs pour plus d'informations.")
        return None

    try:
        # Configuration du modèle avec chargement différé
        model_name = "google/gemma-3n-e4b-it"
        
        # Chargement progressif
        progress_bar = st.progress(0)
        
        def progress_callback(step, total_steps):
            progress = int((step / total_steps) * 100)
            progress_bar.progress(min(progress, 100))
        
        with st.spinner('Chargement du modèle Gemma 3n...'):
            MODEL = pipeline(
                "image-text-to-text",
                model=model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                model_kwargs={
                    "trust_remote_code": True,
                    "token": os.environ["HF_TOKEN"],
                    "cache_dir": "./model_cache"
                },
                callback=progress_callback
            )
            
        progress_bar.empty()
        return MODEL
        
    except Exception as e:
        import traceback
        logger.error(f"Erreur lors du chargement du modèle : {str(e)}")
        traceback.print_exc()
        st.error("Erreur lors du chargement du modèle. Vérifiez les logs pour plus d'informations.")
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

def health_check():
    """Endpoint de santé pour vérifier que l'application est en cours d'exécution"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": time.time()
    }

def main():
    # Vérification de l'endpoint de santé
    if "health" in st.experimental_get_query_params():
        st.json(health_check())
        st.stop()
    
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
    
    # Chargement du modèle avec gestion d'erreur améliorée
    with st.spinner("Initialisation de l'application..."):
        model = load_model()
    
    if model is None:
        st.error("""
        ❌ Impossible de charger le modèle. Vérifiez que :
        - Vous êtes connecté à Internet
        - Votre token d'API Hugging Face est valide (variable d'environnement HF_TOKEN)
        - Vous avez accepté les conditions d'utilisation du modèle Gemma 3n
        - Vous avez suffisamment de mémoire GPU disponible
        
        Essayez de rafraîchir la page dans quelques instants.
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
                try:
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
                except Exception as e:
                    import traceback
                    logger.error(f"Erreur lors de l'analyse : {str(e)}")
                    traceback.print_exc()
                    st.error("Une erreur est survenue lors de l'analyse. Veuillez réessayer ou contacter le support.")
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

def run_fastapi():
    """Lance le serveur FastAPI pour les endpoints d'API"""
    app = FastAPI()
    
    @app.get("/health")
    async def health():
        return health_check()
    
    uvicorn.run(app, host="0.0.0.0", port=8501)

if __name__ == "__main__":
    # Si l'argument --api est passé, on lance le serveur FastAPI
    import sys
    if "--api" in sys.argv:
        run_fastapi()
    else:
        # Sinon, on lance l'interface Streamlit
        main()
