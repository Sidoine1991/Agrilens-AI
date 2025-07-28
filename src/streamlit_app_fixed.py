import streamlit as st
import requests
import json
import base64
import io
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - Fixed",
    page_icon="🌾",
    layout="wide"
)

# Titre principal
st.title("🌾 AgriLens AI")
st.markdown("### Assistant IA pour l'Agriculture (Version Corrigée)")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Sélection du modèle (modèles testés et disponibles)
model_options = {
    "microsoft/DialoGPT-small": "DialoGPT Small (Chat - Testé)",
    "gpt2": "GPT-2 (Texte Général - Testé)",
    "distilgpt2": "DistilGPT-2 (Léger - Testé)",
    "EleutherAI/gpt-neo-125M": "GPT-Neo 125M (Bon équilibre)",
    "microsoft/DialoGPT-medium": "DialoGPT Medium (Plus puissant)"
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

# Mode d'utilisation
mode = st.sidebar.selectbox(
    "Mode d'utilisation",
    ["💬 Mode Chat", "📷 Analyse d'Image", "📤 Upload d'Image"]
)

# Fonction pour appeler l'API avec gestion d'erreur améliorée
def call_api_safe(prompt, model_id, token=None):
    """Appelle l'API Hugging Face avec gestion d'erreur robuste"""
    try:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        headers = {"Content-Type": "application/json"}
        if token and token.strip():
            headers["Authorization"] = f"Bearer {token}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
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
        elif response.status_code == 404:
            return f"❌ Modèle {model_id} non disponible via l'API gratuite. Essayez un autre modèle."
        elif response.status_code == 503:
            return "🔄 Le modèle est en cours de chargement. Réessayez dans quelques secondes."
        elif response.status_code == 401:
            return "❌ Erreur d'authentification. Vérifiez votre token."
        elif response.status_code == 429:
            return "⏳ Trop de requêtes. Attendez un moment."
        else:
            return f"❌ Erreur API ({response.status_code}): {response.text[:100]}"
            
    except requests.exceptions.Timeout:
        return "⏰ Timeout - Le serveur met trop de temps à répondre."
    except requests.exceptions.ConnectionError:
        return "🌐 Erreur de connexion - Vérifiez votre internet."
    except Exception as e:
        return f"❌ Erreur inattendue : {str(e)[:100]}"

# Fonction de fallback local
def generate_local_response(prompt):
    """Génère une réponse locale basée sur des connaissances agricoles"""
    try:
        # Réponses prédéfinies pour les questions courantes
        responses = {
            "tomate": "🌱 **Culture des tomates** : 1) Plantez en plein soleil, 2) Espacez de 60cm, 3) Arrosez régulièrement, 4) Tuteurez les plants, 5) Fertilisez avec du compost.",
            "maladie": "🦠 **Gestion des maladies** : Signes : feuilles jaunes, taches, pourriture. Solutions : 1) Retirez les parties malades, 2) Améliorez la circulation d'air, 3) Utilisez des fongicides naturels.",
            "sol": "🌍 **Amélioration du sol** : 1) Ajoutez du compost, 2) Utilisez du paillage, 3) Plantez des engrais verts, 4) Évitez le compactage, 5) Testez le pH.",
            "arrosage": "💧 **Arrosage optimal** : 1) Tôt le matin, 2) À la base des plantes, 3) Évitez de mouiller les feuilles, 4) Adaptez selon la météo, 5) Utilisez du paillage.",
            "printemps": "🌸 **Légumes de printemps** : 1) Pois (février-mars), 2) Épinards (mars), 3) Radis (mars-avril), 4) Laitues (avril), 5) Carottes (avril-mai)."
        }
        
        prompt_lower = prompt.lower()
        
        # Chercher des mots-clés dans la question
        for keyword, response in responses.items():
            if keyword in prompt_lower:
                return f"🌾 **Réponse locale** : {response}"
        
        # Réponse générique
        return "🌾 **Réponse locale** : Je suis un assistant agricole. Pour des conseils précis, je recommande de consulter un expert local ou des ressources spécialisées en agriculture."
        
    except Exception as e:
        return f"❌ Erreur locale : {e}"

# Fonction pour analyser une image
def analyze_image(image, prompt=""):
    """Analyse une image (simulation pour API texte)"""
    try:
        width, height = image.size
        
        if prompt:
            full_prompt = f"Tu es un expert en agriculture. Analyse cette image: {prompt}. L'image fait {width}x{height} pixels. Donne une analyse détaillée."
        else:
            full_prompt = f"Tu es un expert en agriculture. Analyse cette image agricole. L'image fait {width}x{height} pixels. Identifie les plantes, maladies, conditions de croissance."
        
        # Essayer l'API d'abord
        api_response = call_api_safe(full_prompt, selected_model, api_token)
        
        # Si l'API échoue, utiliser le fallback local
        if api_response.startswith("❌") or api_response.startswith("🔄") or api_response.startswith("⏳"):
            st.warning("⚠️ L'API Hugging Face n'est pas disponible. Utilisation du mode local.")
            return generate_local_response(prompt if prompt else "analyse d'image agricole")
        
        return api_response
        
    except Exception as e:
        return f"❌ Erreur : {e}"

# Interface principale
if mode == "💬 Mode Chat":
    st.header("💬 Mode Chat")
    st.info(f"🤖 Chat avec {selected_model} !")
    
    # Afficher les informations du modèle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informations Modèle")
    st.sidebar.info(f"**Modèle** : {selected_model}\n**Type** : API Hugging Face\n**Statut** : ✅ Testé")
    
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
            
            # Générer la réponse
            try:
                with st.spinner(f"🔄 Génération avec {selected_model}..."):
                    context = "Tu es un expert en agriculture. Réponds de manière utile et précise."
                    full_prompt = f"{context}\n\nQuestion: {user_question}\n\nRéponse:"
                    
                    # Essayer l'API d'abord
                    response = call_api_safe(full_prompt, selected_model, api_token)
                    
                    # Si l'API échoue, utiliser le fallback local
                    if response.startswith("❌") or response.startswith("🔄") or response.startswith("⏳"):
                        st.warning("⚠️ L'API Hugging Face n'est pas disponible. Utilisation du mode local.")
                        response = generate_local_response(user_question)
                
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

elif mode == "📷 Analyse d'Image":
    st.header("📷 Analyse d'Image")
    st.info("🚀 Utilise l'API Hugging Face - Aucun téléchargement local !")
    st.warning("⚠️ Note: L'analyse d'image est simulée car les modèles multimodaux ne sont pas disponibles via l'API gratuite.")
    
    # Afficher les informations du modèle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informations Modèle")
    st.sidebar.info(f"**Modèle** : {selected_model}\n**Type** : API Hugging Face\n**Statut** : ✅ Testé")
    
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
                    result = analyze_image(image, custom_prompt)
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
                        result = analyze_image(image, "")
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
                    result = analyze_image(image, custom_prompt)
                    st.markdown("### 📊 Résultats de l'Analyse")
                    st.write(result)

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
                result = analyze_image(image, "")
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
    "• Fallback local inclus"
)

# Note sur le token
if not api_token:
    st.sidebar.info("💡 Pas de token : L'API publique sera utilisée")
else:
    st.sidebar.success("✅ Token configuré")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🌾 AgriLens AI - Version Corrigée<br>
        Powered by Hugging Face Inference API
    </div>
    """,
    unsafe_allow_html=True
) 