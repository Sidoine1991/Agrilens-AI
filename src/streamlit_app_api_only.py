import streamlit as st
import requests
import json
import base64
import io
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - API Only",
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
    "distilgpt2": "DistilGPT-2 (Léger)"
}

selected_model = st.sidebar.selectbox(
    "🤖 Modèle à utiliser",
    list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=0
)

# Token API (optionnel)
api_token = st.sidebar.text_input(
    "🔑 Token Hugging Face (optionnel)",
    type="password",
    help="Améliore les performances"
)

# Fonction pour appeler l'API
def call_api(prompt, model_id, token=None):
    """Appelle l'API Hugging Face Inference"""
    try:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        headers = {"Content-Type": "application/json"}
        if token and token.strip():
            headers["Authorization"] = f"Bearer {token}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "do_sample": True
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
            return "❌ Erreur d'authentification. Vérifiez votre token."
        elif response.status_code == 429:
            return "⏳ Trop de requêtes. Attendez un moment."
        else:
            return f"❌ Erreur API ({response.status_code})"
            
    except Exception as e:
        return f"❌ Erreur de connexion : {e}"

# Fonction pour analyser une image
def analyze_image(image, prompt=""):
    """Analyse une image (simulation pour API texte)"""
    try:
        width, height = image.size
        
        if prompt:
            full_prompt = f"Tu es un expert en agriculture. Analyse cette image: {prompt}. L'image fait {width}x{height} pixels. Donne une analyse détaillée."
        else:
            full_prompt = f"Tu es un expert en agriculture. Analyse cette image agricole. L'image fait {width}x{height} pixels. Identifie les plantes, maladies, conditions de croissance."
        
        return call_api(full_prompt, selected_model, api_token)
        
    except Exception as e:
        return f"❌ Erreur : {e}"

# Interface principale
st.header("💬 Chat Agricole")
st.info("🚀 Utilise l'API Hugging Face - Aucun téléchargement !")

# Historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
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
        
        # Générer la réponse
        with st.spinner(f"🔄 Génération avec {selected_model}..."):
            context = "Tu es un expert en agriculture. Réponds de manière utile et précise."
            full_prompt = f"{context}\n\nQuestion: {user_question}\n\nRéponse:"
            response = call_api(full_prompt, selected_model, api_token)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    else:
        st.warning("⚠️ Veuillez saisir une question.")

# Bouton pour effacer l'historique
if st.button("🗑️ Effacer l'historique"):
    st.session_state.messages = []
    st.rerun()

# Section analyse d'image
st.markdown("---")
st.header("📷 Analyse d'Image")

uploaded_file = st.file_uploader(
    "Choisissez une image agricole...",
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image à analyser", use_container_width=True)
    
    custom_prompt = st.text_area(
        "💭 Question spécifique (optionnel)",
        placeholder="Ex: Identifie les maladies présentes"
    )
    
    if st.button("🔍 Analyser", type="primary"):
        with st.spinner(f"🔄 Analyse avec {selected_model}..."):
            result = analyze_image(image, custom_prompt)
            st.markdown("### 📊 Résultats")
            st.write(result)

# Informations
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ API Hugging Face")
st.sidebar.info(
    "**Avantages** :\n"
    "• Aucun téléchargement\n"
    "• Pas de GPU requis\n"
    "• Mise à jour automatique"
)

if not api_token:
    st.sidebar.info("💡 Pas de token : API publique")
else:
    st.sidebar.success("✅ Token configuré") 