import streamlit as st
import requests
import json
import base64
import io
from PIL import Image
import random
from transformers import pipeline
import torch
import traceback

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - Diagnostic des Plantes",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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

# Titre principal
st.title("🌾 AgriLens AI")
st.markdown("### Assistant IA pour l'Agriculture (Modèles Locaux Légers)")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Sélection du modèle
model_options = {
    "local_simple": "Assistant Local Simple (Recommandé)",
    "local_advanced": "Assistant Local Avancé",
    "api_fallback": "API Hugging Face (Fallback)"
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
    help="Pour l'API Hugging Face uniquement"
)

# Mode d'utilisation
mode = st.sidebar.selectbox(
    "Mode d'utilisation",
    ["📷 Analyse d'Image", "💬 Mode Chat", "📤 Upload d'Image"]
)

# Base de connaissances agricoles locale
AGRICULTURE_KNOWLEDGE = {
    "cultures": {
        "tomate": {
            "plantation": "Plantez les tomates en plein soleil, espacées de 60cm",
            "arrosage": "Arrosez régulièrement à la base, évitez de mouiller les feuilles",
            "entretien": "Tuteurez les plants, supprimez les gourmands",
            "maladies": "Mildiou, alternariose, oïdium",
            "solutions": "Rotation des cultures, fongicides naturels, bonne aération"
        },
        "salade": {
            "plantation": "Semez en place ou repiquez, espacement 25cm",
            "arrosage": "Arrosage régulier mais modéré",
            "entretien": "Binage régulier, paillage",
            "maladies": "Mildiou, pourriture grise",
            "solutions": "Éviter l'excès d'humidité, rotation"
        },
        "carotte": {
            "plantation": "Semez en place, sol meuble et profond",
            "arrosage": "Arrosage régulier pour éviter la fourchure",
            "entretien": "Éclaircissage, binage",
            "maladies": "Alternariose, pourriture",
            "solutions": "Rotation, sol bien drainé"
        }
    },
    "maladies": {
        "mildiou": "Champignon qui attaque feuilles et fruits. Solutions : fongicides, bonne aération, rotation",
        "oïdium": "Poudre blanche sur les feuilles. Solutions : soufre, bicarbonate, variétés résistantes",
        "pourriture": "Pourriture des racines ou fruits. Solutions : drainage, rotation, fongicides"
    },
    "techniques": {
        "rotation": "Changez l'emplacement des cultures chaque année pour éviter les maladies",
        "paillage": "Couvrez le sol avec de la paille ou du compost pour retenir l'humidité",
        "compost": "Fertilisez avec du compost fait maison pour enrichir le sol",
        "binage": "Aérez le sol en surface pour favoriser la croissance des racines"
    }
}

# Charger le modèle local Gemma une seule fois
@st.cache_resource(show_spinner=False)
def get_local_gemma():
    try:
        st.info("Chargement du modèle local Gemma depuis D:/Dev/model_gemma...")
        generator = pipeline("text-generation", model="D:/Dev/model_gemma")
        st.success(f"Modèle Gemma chargé : {generator}")
        return generator
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle local Gemma : {e}")
        st.error(traceback.format_exc())
        return None

# Charger le modèle multimodal Gemma 3n
@st.cache_resource(show_spinner=False)
def get_local_gemma_multimodal():
    try:
        st.info("Chargement du modèle multimodal Gemma 3n depuis D:/Dev/model_gemma...")
        pipe = pipeline(
            "image-text-to-text",
            model="D:/Dev/model_gemma",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        st.success(f"Modèle Gemma multimodal chargé : {pipe}")
        return pipe
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle Gemma multimodal : {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Fonction pour analyser le texte avec le modèle local Gemma
def analyze_text_local(prompt):
    """Analyse le texte avec le modèle local Gemma"""
    try:
        generator = get_local_gemma()
        result = generator(prompt, max_new_tokens=200)
        response = result[0]['generated_text']
        return response
    except Exception as e:
        return f"❌ Erreur d'analyse locale : {e}"

# Fonction pour appeler l'API Hugging Face (fallback)
def call_api_fallback(prompt, token=None):
    """Appelle l'API Hugging Face comme fallback"""
    try:
        url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        
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
                return result[0].get("generated_text", "")
            return str(result)
        else:
            return None
            
    except Exception as e:
        return None

# --- Sidebar : Chargement du modèle multimodal ---
with st.sidebar:
    st.header("Réglages IA locale")
    if "gemma_model" not in st.session_state:
        st.session_state.gemma_model = None
        st.session_state.gemma_model_status = "Modèle non chargé"

    if st.button("Charger le modèle Gemma 3n multimodal (CPU)"):
        with st.spinner("Chargement du modèle (CPU)..."):
            try:
                st.session_state.gemma_model = pipeline(
                    "image-text-to-text",
                    model="D:/Dev/model_gemma",
                    device=-1,  # Forcer le CPU
                    torch_dtype=torch.float32,
                )
                st.session_state.gemma_model_status = "Modèle chargé et prêt (CPU)"
                st.success("Modèle Gemma 3n multimodal chargé avec succès sur CPU !")
            except Exception as e:
                st.session_state.gemma_model = None
                st.session_state.gemma_model_status = f"Erreur : {e}"
                st.error(f"Erreur lors du chargement du modèle : {e}")
                st.error(traceback.format_exc())
    st.info(f"État du modèle : {st.session_state.gemma_model_status}")

# --- Fonction d'analyse d'image ---
def analyze_image_local(image, prompt=""):
    if st.session_state.gemma_model is None:
        return "❌ Modèle non chargé. Veuillez le charger dans les réglages."
    try:
        pipe = st.session_state.gemma_model
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Tu es un assistant agricole expert en diagnostic de maladies de plantes. Donne des réponses claires et structurées."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt or "Décris la maladie présente sur cette plante et donne des recommandations pratiques."}
                ]
            }
        ]
        
        # Paramètres de génération améliorés
        result = pipe(
            text=messages, 
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        if isinstance(result, list) and "generated_text" in result[0]:
            response = result[0]["generated_text"]
            
            # Nettoyage de la réponse
            if response and len(response.strip()) > 0:
                # Vérifier si la réponse contient déjà des recommandations
                if "recommandation" not in response.lower() and "action" not in response.lower():
                    response += "\n\n**Recommandations ou actions urgentes :**\n• Isolez la plante malade si possible\n• Appliquez un traitement adapté\n• Surveillez les autres plantes\n• Consultez un expert si nécessaire"
                
                return response
            else:
                return "❌ Erreur : Réponse vide du modèle"
        
        return str(result)
    except Exception as e:
        return f"❌ Erreur lors de l'analyse d'image : {e}"

# Interface principale
if mode == "📷 Analyse d'Image":
    st.header("📷 Analyse d'Image")
    st.info("🤖 Utilise l'IA locale pour analyser vos images agricoles !")
    
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
            
            if st.button("🔍 Analyser avec l'IA Locale", type="primary", disabled=st.session_state.gemma_model is None):
                with st.spinner("Analyse de l'image en cours... Merci de patienter, cela peut prendre plusieurs dizaines de secondes."):
                    result = analyze_image_local(image, custom_prompt)
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
                    
                    result = analyze_image_local(image, "")
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
            
            if st.button("🔍 Analyser avec l'IA Locale", type="primary", disabled=st.session_state.gemma_model is None):
                with st.spinner("Analyse de l'image en cours... Merci de patienter, cela peut prendre plusieurs dizaines de secondes."):
                    result = analyze_image_local(image, custom_prompt)
                st.markdown("### 📊 Résultats de l'Analyse")
                st.write(result)

elif mode == "💬 Mode Chat":
    st.header("💬 Mode Chat")
    st.info("🤖 Posez des questions sur l'agriculture - IA Locale !")
    
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
            
            # Générer la réponse selon le modèle sélectionné
            try:
                if selected_model == "local_simple":
                    response = analyze_text_local(user_question)
                elif selected_model == "local_advanced":
                    # Version avancée avec plus de détails
                    response = analyze_text_local(user_question) + "\n\n🔬 **Analyse avancée** : Cette réponse est générée par notre IA locale spécialisée en agriculture."
                elif selected_model == "api_fallback":
                    # Essayer l'API d'abord, puis fallback local
                    api_response = call_api_fallback(user_question, api_token)
                    if api_response:
                        response = f"🌐 **Réponse API** : {api_response}"
                    else:
                        response = analyze_text_local(user_question)
                        st.warning("⚠️ API non disponible, utilisation du mode local.")
                else:
                    response = analyze_text_local(user_question)
                
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
    st.info("🎯 Version simplifiée avec IA locale")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image agricole...",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image à analyser", use_container_width=True)
        
        if st.button("🚀 Analyser avec l'IA Locale", type="primary", disabled=st.session_state.gemma_model is None):
            with st.spinner("Analyse de l'image en cours... Merci de patienter, cela peut prendre plusieurs dizaines de secondes."):
                result = analyze_image_local(image, "")
            st.markdown("### 📊 Analyse")
            st.write(result)

# Informations sur l'IA locale
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Informations")
st.sidebar.info(
    "**Avantages de l'IA Locale :**\n"
    "✅ Fonctionne hors ligne\n"
    "✅ Réponses instantanées\n"
    "✅ Connaissances agricoles spécialisées\n"
    "✅ Pas de limitations API"
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🌾 AgriLens AI - IA Locale<br>
        Powered by Local AI Models
    </div>
    """,
    unsafe_allow_html=True
) 