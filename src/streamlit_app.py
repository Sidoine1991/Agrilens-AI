import streamlit as st
from PIL import Image
from transformers import pipeline
import torch
import os

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - Diagnostic des Plantes",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Charge le modèle Gemma 3n"""
    try:
        # Vérification du token d'accès
        if "HF_TOKEN" not in os.environ:
            st.error("Token d'accès Hugging Face manquant. Veuillez configurer le token HF_TOKEN dans les paramètres du Space.")
            return None
            
        pipe = pipeline(
            "image-text-to-text",
            model="google/gemma-3n-e4b-it",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            model_kwargs={
                "trust_remote_code": True,
                "token": os.environ["HF_TOKEN"]
            }
        )
        return pipe
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

def main():
    st.title("🌱 AgriLens AI - Diagnostic des Plantes")
    st.markdown("### Téléchargez une photo de plante pour analyse")

    # Chargement du modèle
    with st.spinner('Chargement du modèle Gemma 3n...'):
        model = load_model()
    
    if model is None:
        st.error("Impossible de charger le modèle. Vérifiez les logs pour plus d'informations.")
        return

    # Téléchargement de l'image
    uploaded_file = st.file_uploader(
        "Choisissez une image de plante...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Affichage de l'image
        image = Image.open(uploaded_file)
        st.image(image, caption='Image téléchargée', use_column_width=True)
        
        if st.button("Analyser l'image"):
            with st.spinner('Analyse en cours...'):
                try:
                    prompt = """Analyse cette image de plante et identifie les maladies potentielles.
                    Fournis une réponse structurée avec :
                    1. Le nom de la plante (si identifiable)
                    2. Les maladies ou problèmes détectés
                    3. Le niveau de confiance
                    4. Des recommandations de traitement
                    """
                    
                    # Appel au modèle
                    response = model(image, prompt=prompt, max_new_tokens=500)
                    
                    # Affichage des résultats
                    st.markdown("### 🔍 Résultats de l'analyse")
                    st.markdown(response[0]['generated_text'])
                    
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {str(e)}")

if __name__ == "__main__":
    main()