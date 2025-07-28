import streamlit as st
import requests
import json
import base64
import io
from PIL import Image
import time

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - Inference API",
    page_icon="🌾",
    layout="wide"
)

# Titre principal
st.title("🌾 AgriLens AI")
st.markdown("### Assistant IA pour l'Agriculture (API Hugging Face)")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Sélection du modèle
model_options = {
    "microsoft/DialoGPT-medium": "DialoGPT Medium (Chat - Recommandé)",
    "microsoft/DialoGPT-small": "DialoGPT Small (Chat Rapide)",
    "gpt2": "GPT-2 (Texte Général)",
    "distilgpt2": "DistilGPT-2 (Léger)",
    "EleutherAI/gpt-neo-125M": "GPT-Neo 125M (Bon équilibre)",
    "microsoft/DialoGPT-large": "DialoGPT Large (Plus puissant)"
}

selected_model = st.sidebar.selectbox(
    "🤖 Modèle à utiliser",
    list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=0
)

# Token API (recommandé)
api_token = st.sidebar.text_input(
    "🔑 Token Hugging Face (recommandé)",
    type="password",
    help="Améliore les performances et évite les limitations"
)

# Mode d'utilisation
mode = st.sidebar.selectbox(
    "Mode d'utilisation",
    ["📷 Analyse d'Image", "💬 Mode Chat", "📤 Upload d'Image"]
)

# Fonction pour appeler l'API Hugging Face
def call_huggingface_api(prompt, model_id, token=None, max_length=200):
    """Appelle l'API Hugging Face Inference"""
    try:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        headers = {"Content-Type": "application/json"}
        if token and token.strip():
            headers["Authorization"] = f"Bearer {token}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", str(result[0]))
            return str(result)
        elif response.status_code == 503:
            return "🔄 Le modèle est en cours de chargement. Réessayez dans quelques secondes."
        elif response.status_code == 401:
            return "❌ Erreur d'authentification. Vérifiez votre token Hugging Face."
        elif response.status_code == 429:
            return "⏳ Trop de requêtes. Attendez un moment avant de réessayer."
        else:
            return f"❌ Erreur API ({response.status_code}): {response.text}"
            
    except Exception as e:
        return f"❌ Erreur de connexion : {e}"

# Fonction pour analyser une image via API
def analyze_image_api(image, prompt="", model_id="microsoft/DialoGPT-medium", token=None):
    """Analyse une image via l'API (simulation pour modèles texte uniquement)"""
    try:
        # Convertir l'image en RGB si nécessaire
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Obtenir les dimensions de l'image
        width, height = image.size
        
        # Préparer le prompt pour l'analyse simulée
        if prompt:
            full_prompt = f"Tu es un expert en agriculture. Analyse cette image agricole: {prompt}. L'image fait {width}x{height} pixels. Donne une analyse détaillée et des recommandations."
        else:
            full_prompt = f"Tu es un expert en agriculture. Analyse cette image agricole en détail. L'image fait {width}x{height} pixels. Identifie les plantes, les maladies, les conditions de croissance, et donne des recommandations précises."
        
        return call_huggingface_api(full_prompt, model_id, token)
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse : {e}"

# Fonction pour le chat via API
def chat_with_api(prompt, model_id, token=None):
    """Chat avec l'API Hugging Face"""
    try:
        # Préparer le contexte agricole
        context = "Tu es un expert en agriculture. Réponds de manière utile et précise."
        full_prompt = f"{context}\n\nQuestion: {prompt}\n\nRéponse:"
        
        return call_huggingface_api(full_prompt, model_id, token)
        
    except Exception as e:
        return f"❌ Erreur de chat : {e}"

# Interface principale
if mode == "📷 Analyse d'Image":
    st.header("📷 Analyse d'Image")
    st.info("🚀 Utilise l'API Hugging Face - Aucun téléchargement local !")
    st.warning("⚠️ Note: L'analyse d'image est simulée car les modèles multimodaux ne sont pas disponibles via l'API gratuite.")
    
    # Afficher les informations du modèle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informations Modèle")
    st.sidebar.info(f"**Modèle** : {selected_model}\n**Type** : API Hugging Face\n**Statut** : ✅ Disponible")
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "🌐 URL", "📸 Webcam"])
    
    with tab1:
        st.subheader("📤 Upload d'Image")
        uploaded_file = st.file_uploader(
            "Choisissez une image...",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image uploadée", use_container_width=True)
            
            # Prompt personnalisé
            custom_prompt = st.text_area(
                "💭 Question spécifique (optionnel)",
                placeholder="Ex: Identifie les maladies présentes sur ces feuilles"
            )
            
            if st.button("🔍 Analyser avec l'API", type="primary"):
                with st.spinner(f"🔄 Analyse avec {selected_model}..."):
                    result = analyze_image_api(image, custom_prompt, selected_model, api_token)
                    st.markdown("### 📊 Résultats de l'Analyse")
                    st.write(result)
    
    with tab2:
        st.subheader("🌐 Image depuis URL")
        url = st.text_input(
            "Entrez l'URL de l'image",
            placeholder="https://example.com/image.jpg"
        )
        
        if url:
            if st.button("📥 Télécharger et Analyser", type="primary"):
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                    st.image(image, caption="Image téléchargée", use_container_width=True)
                    
                    with st.spinner(f"🔄 Analyse avec {selected_model}..."):
                        result = analyze_image_api(image, "", selected_model, api_token)
                        st.markdown("### 📊 Résultats de l'Analyse")
                        st.write(result)
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
    
    with tab3:
        st.subheader("📸 Capture Webcam")
        camera_input = st.camera_input("📸 Prenez une photo")
        
        if camera_input is not None:
            image = Image.open(camera_input)
            st.image(image, caption="Image capturée", use_container_width=True)
            
            custom_prompt = st.text_area(
                "💭 Question spécifique (optionnel)",
                placeholder="Ex: Identifie les maladies présentes sur ces feuilles"
            )
            
            if st.button("🔍 Analyser avec l'API", type="primary"):
                with st.spinner(f"🔄 Analyse avec {selected_model}..."):
                    result = analyze_image_api(image, custom_prompt, selected_model, api_token)
                    st.markdown("### 📊 Résultats de l'Analyse")
                    st.write(result)

elif mode == "💬 Mode Chat":
    st.header("💬 Mode Chat")
    st.info(f"🤖 Chat avec l'API {selected_model} !")
    
    # Afficher les informations du modèle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informations Modèle")
    st.sidebar.info(f"**Modèle** : {selected_model}\n**Type** : API Hugging Face\n**Statut** : ✅ Disponible")
    
    # Historique des messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Afficher l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Zone de saisie
    st.markdown("---")
    st.subheader("💭 Posez votre question")
    
    user_question = st.text_input(
        "Votre question sur l'agriculture :",
        placeholder="Ex: Comment cultiver des tomates ?",
        key="chat_input"
    )
    
    # Bouton pour envoyer
    if st.button("🚀 Envoyer", type="primary"):
        if user_question.strip():
            # Ajouter le message utilisateur
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # Générer la réponse via API
            try:
                with st.spinner(f"🔄 Génération avec {selected_model}..."):
                    response = chat_with_api(user_question, selected_model, api_token)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
                
            except Exception as e:
                error_msg = f"❌ Erreur : {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.rerun()
        else:
            st.warning("⚠️ Veuillez saisir une question.")
    
    # Bouton pour effacer l'historique
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

elif mode == "📤 Upload d'Image":
    st.header("📤 Upload d'Image Simple")
    st.info(f"🎯 Version simplifiée avec {selected_model}")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image agricole...",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image à analyser", use_container_width=True)
        
        if st.button("🚀 Analyser avec l'API", type="primary"):
            with st.spinner(f"🔄 Analyse avec {selected_model}..."):
                result = analyze_image_api(image, "", selected_model, api_token)
                st.markdown("### 📊 Analyse")
                st.write(result)

# Informations sur l'API
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ API Hugging Face")
st.sidebar.info(
    "**Avantages** :\n"
    "• Aucun téléchargement\n"
    "• Pas de GPU requis\n"
    "• Mise à jour automatique\n"
    "• Modèles premium disponibles"
)

# Note sur le token
if not api_token:
    st.sidebar.info("💡 Pas de token : L'API publique sera utilisée (limitations possibles)")
else:
    st.sidebar.success("✅ Token configuré")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🌾 AgriLens AI - Inference API<br>
        Powered by Hugging Face Inference API
    </div>
    """,
    unsafe_allow_html=True
) 