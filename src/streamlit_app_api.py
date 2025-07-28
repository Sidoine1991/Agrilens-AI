import streamlit as st
import os
import sys
import base64
import io
from PIL import Image
import requests
import json

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - Analyse d'Images Agricoles",
    page_icon="🌾",
    layout="wide"
)

# Titre principal
st.title("🌾 AgriLens AI")
st.markdown("### Assistant IA pour l'Analyse d'Images Agricoles (API Version)")

# Sidebar pour les options
st.sidebar.header("⚙️ Configuration")

# Configuration de l'API
API_URL = "https://api-inference.huggingface.co/models/google/gemma-3n-E4B-it"
API_TOKEN = st.sidebar.text_input(
    "🔑 Token Hugging Face (optionnel)",
    type="password",
    help="Votre token Hugging Face pour de meilleures performances"
)

# Mode d'utilisation
mode = st.sidebar.selectbox(
    "Mode d'utilisation",
    ["📷 Analyse d'Image", "💬 Mode Chat", "📁 Import Local"]
)

# Fonction pour appeler l'API Hugging Face
def call_huggingface_api(messages, api_token=None):
    """Appelle l'API Hugging Face pour la génération de texte"""
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        
        payload = {
            "inputs": messages,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            return str(result)
        else:
            return f"❌ Erreur API ({response.status_code}): {response.text}"
            
    except Exception as e:
        return f"❌ Erreur lors de l'appel API : {e}"

# Fonction pour analyser une image
def analyze_image(image, prompt="", api_token=None):
    """Analyse une image avec l'API Hugging Face"""
    try:
        # Convertir l'image en base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Préparer les messages pour le modèle
        if prompt:
            user_content = [
                {"type": "image", "image": f"data:image/jpeg;base64,{img_str}"},
                {"type": "text", "text": prompt}
            ]
        else:
            user_content = [
                {"type": "image", "image": f"data:image/jpeg;base64,{img_str}"},
                {"type": "text", "text": "Analyse cette image agricole en détail. Identifie les plantes, les maladies, les conditions de croissance, et donne des recommandations."}
            ]
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Tu es un expert en agriculture et en analyse d'images agricoles. Tu peux identifier les plantes, les maladies, les conditions de croissance et donner des recommandations précises."}]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        return call_huggingface_api(messages, api_token)
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse : {e}"

# Fonction pour traiter une image depuis une URL
def process_image_from_url(url):
    """Télécharge et traite une image depuis une URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"❌ Erreur lors du téléchargement de l'image : {e}")
        return None

# Fonction pour traiter une image depuis un fichier uploadé
def process_uploaded_image(uploaded_file):
    """Traite une image uploadée"""
    try:
        image = Image.open(uploaded_file)
        return image
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement de l'image : {e}")
        return None

# Fonction pour traiter une image depuis une webcam
def process_webcam_image():
    """Traite une image depuis la webcam"""
    try:
        camera_input = st.camera_input("📸 Prenez une photo")
        if camera_input is not None:
            image = Image.open(camera_input)
            return image
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors de la capture webcam : {e}")
        return None

# Interface principale selon le mode
if mode == "📷 Analyse d'Image":
    st.header("📷 Analyse d'Image")
    
    # Onglets pour les différentes méthodes d'import
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "🌐 URL", "📸 Webcam", "📋 Base64"])
    
    with tab1:
        st.subheader("📤 Upload d'Image")
        uploaded_file = st.file_uploader(
            "Choisissez une image...",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Formats supportés : PNG, JPG, JPEG, GIF, BMP"
        )
        
        if uploaded_file is not None:
            image = process_uploaded_image(uploaded_file)
            if image:
                st.image(image, caption="Image uploadée", use_column_width=True)
                
                # Prompt personnalisé
                custom_prompt = st.text_area(
                    "💭 Question ou instruction spécifique (optionnel)",
                    placeholder="Ex: Identifie les maladies présentes sur ces feuilles",
                    help="Laissez vide pour une analyse générale"
                )
                
                if st.button("🔍 Analyser l'Image", type="primary"):
                    with st.spinner("🔄 Analyse en cours via API..."):
                        result = analyze_image(image, custom_prompt, API_TOKEN)
                        st.markdown("### 📊 Résultats de l'Analyse")
                        st.write(result)
    
    with tab2:
        st.subheader("🌐 Image depuis URL")
        url = st.text_input(
            "Entrez l'URL de l'image",
            placeholder="https://example.com/image.jpg",
            help="URL directe vers une image"
        )
        
        if url:
            if st.button("📥 Télécharger et Analyser", type="primary"):
                with st.spinner("🔄 Téléchargement et analyse..."):
                    image = process_image_from_url(url)
                    if image:
                        st.image(image, caption="Image téléchargée", use_column_width=True)
                        result = analyze_image(image, "", API_TOKEN)
                        st.markdown("### 📊 Résultats de l'Analyse")
                        st.write(result)
    
    with tab3:
        st.subheader("📸 Capture Webcam")
        st.info("📱 Utilisez votre webcam pour capturer une image")
        
        image = process_webcam_image()
        if image:
            st.image(image, caption="Image capturée", use_column_width=True)
            
            custom_prompt = st.text_area(
                "💭 Question ou instruction spécifique (optionnel)",
                placeholder="Ex: Identifie les maladies présentes sur ces feuilles",
                help="Laissez vide pour une analyse générale"
            )
            
            if st.button("🔍 Analyser l'Image", type="primary"):
                with st.spinner("🔄 Analyse en cours..."):
                    result = analyze_image(image, custom_prompt, API_TOKEN)
                    st.markdown("### 📊 Résultats de l'Analyse")
                    st.write(result)
    
    with tab4:
        st.subheader("📋 Image en Base64")
        st.info("🔧 Pour les développeurs : collez une image encodée en base64")
        
        base64_input = st.text_area(
            "Collez votre image en base64",
            placeholder="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
            help="Format attendu : data:image/type;base64,données"
        )
        
        if base64_input and st.button("🔍 Analyser l'Image", type="primary"):
            try:
                # Extraire les données base64
                if base64_input.startswith('data:image/'):
                    header, data = base64_input.split(',', 1)
                    image_data = base64.b64decode(data)
                    image = Image.open(io.BytesIO(image_data))
                    
                    st.image(image, caption="Image décodée", use_column_width=True)
                    
                    with st.spinner("🔄 Analyse en cours..."):
                        result = analyze_image(image, "", API_TOKEN)
                        st.markdown("### 📊 Résultats de l'Analyse")
                        st.write(result)
                else:
                    st.error("❌ Format base64 invalide. Utilisez le format : data:image/type;base64,données")
            except Exception as e:
                st.error(f"❌ Erreur lors du décodage : {e}")

elif mode == "💬 Mode Chat":
    st.header("💬 Mode Chat")
    st.info("💡 Posez des questions générales sur l'agriculture ou demandez des conseils")
    
    # Initialiser l'historique des messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Afficher l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input utilisateur
    if prompt := st.chat_input("Posez votre question..."):
        # Ajouter le message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("🔄 Génération de la réponse via API..."):
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "Tu es un expert en agriculture. Tu peux répondre aux questions sur les plantes, les maladies, les techniques agricoles, etc."}]
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        }
                    ]
                    
                    response = call_huggingface_api(messages, API_TOKEN)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ Erreur lors de la génération : {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Bouton pour effacer l'historique
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

elif mode == "📁 Import Local":
    st.header("📁 Import Local")
    st.info("🔧 Importez une image depuis un chemin local sur votre système")
    
    file_path = st.text_input(
        "Chemin vers l'image",
        placeholder="C:/Users/username/Pictures/image.jpg",
        help="Chemin complet vers le fichier image"
    )
    
    if file_path:
        if st.button("📥 Charger et Analyser", type="primary"):
            try:
                if os.path.exists(file_path):
                    image = Image.open(file_path)
                    st.image(image, caption="Image chargée", use_column_width=True)
                    
                    custom_prompt = st.text_area(
                        "💭 Question ou instruction spécifique (optionnel)",
                        placeholder="Ex: Identifie les maladies présentes sur ces feuilles",
                        help="Laissez vide pour une analyse générale"
                    )
                    
                    if st.button("🔍 Analyser l'Image", type="primary"):
                        with st.spinner("🔄 Analyse en cours..."):
                            result = analyze_image(image, custom_prompt, API_TOKEN)
                            st.markdown("### 📊 Résultats de l'Analyse")
                            st.write(result)
                else:
                    st.error("❌ Fichier non trouvé. Vérifiez le chemin.")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement : {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🌾 AgriLens AI - Assistant IA pour l'Agriculture<br>
        Powered by Google Gemma 3n API
    </div>
    """,
    unsafe_allow_html=True
) 