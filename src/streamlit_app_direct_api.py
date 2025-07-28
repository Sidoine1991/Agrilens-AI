import streamlit as st
import requests
import json
import base64
import io
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - API Directe",
    page_icon="🌾",
    layout="wide"
)

# Titre principal
st.title("🌾 AgriLens AI")
st.markdown("### Assistant IA pour l'Agriculture (API Directe - Aucun Téléchargement)")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Sélection du modèle
model_options = {
    "microsoft/DialoGPT-medium": "DialoGPT Medium (Texte uniquement - Recommandé)",
    "gpt2": "GPT-2 (Texte uniquement)",
    "microsoft/DialoGPT-small": "DialoGPT Small (Plus rapide)",
    "distilgpt2": "DistilGPT-2 (Plus léger)",
    "EleutherAI/gpt-neo-125M": "GPT-Neo 125M (Bon équilibre)"
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
    help="Améliore les performances et évite les limitations. Laissez vide pour utiliser l'API publique."
)

# Note sur le token
if not api_token:
    st.sidebar.info("💡 Pas de token : L'API publique sera utilisée (limitations possibles)")
else:
    st.sidebar.success("✅ Token configuré")

# Mode d'utilisation
mode = st.sidebar.selectbox(
    "Mode d'utilisation",
    ["📷 Analyse d'Image", "💬 Mode Chat", "📤 Upload d'Image"]
)

# Fonction pour appeler l'API
def call_api(prompt, model_id, token=None):
    """Appelle l'API Hugging Face"""
    try:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        headers = {"Content-Type": "application/json"}
        if token and token.strip():
            headers["Authorization"] = f"Bearer {token}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        with st.spinner("🔄 Appel de l'API en cours..."):
            response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            return str(result)
        elif response.status_code == 401:
            return "❌ Erreur d'authentification. Vérifiez votre token Hugging Face ou laissez le champ vide pour utiliser l'API publique."
        elif response.status_code == 503:
            return "🔄 Le modèle est en cours de chargement. Réessayez dans quelques secondes."
        elif response.status_code == 429:
            return "⏳ Trop de requêtes. Attendez un moment avant de réessayer."
        else:
            return f"❌ Erreur API ({response.status_code}): {response.text}"
            
    except Exception as e:
        return f"❌ Erreur de connexion : {e}"

# Fonction de fallback pour générer des réponses locales
def generate_local_response(prompt):
    """Génère une réponse locale basée sur des connaissances agricoles"""
    try:
        # Réponses prédéfinies pour les questions courantes
        responses = {
            "tomate": "Pour cultiver des tomates : 1) Plantez en plein soleil, 2) Espacez de 60cm, 3) Arrosez régulièrement, 4) Tuteurez les plants, 5) Fertilisez avec du compost.",
            "maladie": "Signes de maladies : feuilles jaunes, taches, pourriture. Solutions : 1) Retirez les parties malades, 2) Améliorez la circulation d'air, 3) Utilisez des fongicides naturels.",
            "sol": "Pour améliorer le sol : 1) Ajoutez du compost, 2) Utilisez du paillage, 3) Plantez des engrais verts, 4) Évitez le compactage, 5) Testez le pH.",
            "arrosage": "Arrosage optimal : 1) Tôt le matin, 2) À la base des plantes, 3) Évitez de mouiller les feuilles, 4) Adaptez selon la météo, 5) Utilisez du paillage.",
            "printemps": "Légumes de printemps : 1) Pois (février-mars), 2) Épinards (mars), 3) Radis (mars-avril), 4) Laitues (avril), 5) Carottes (avril-mai)."
        }
        
        prompt_lower = prompt.lower()
        
        # Chercher des mots-clés dans la question
        for keyword, response in responses.items():
            if keyword in prompt_lower:
                return f"🌾 Réponse locale : {response}"
        
        # Réponse générique
        return "🌾 Réponse locale : Je suis un assistant agricole. Pour des conseils précis, je recommande de consulter un expert local ou des ressources spécialisées en agriculture."
        
    except Exception as e:
        return f"❌ Erreur locale : {e}"

# Fonction pour analyser une image
def analyze_image_api(image, prompt="", model_id="microsoft/DialoGPT-medium", token=None):
    """Analyse une image via l'API (simulation pour modèles texte uniquement)"""
    try:
        # Convertir l'image en RGB si elle est en RGBA
        if image.mode == 'RGBA':
            # Créer un fond blanc
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # Utiliser le canal alpha comme masque
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
        
        return call_api(full_prompt, model_id, token)
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse : {e}"

# Interface principale
if mode == "📷 Analyse d'Image":
    st.header("📷 Analyse d'Image")
    st.info("🚀 Utilise directement l'API Hugging Face - Aucun téléchargement !")
    st.warning("⚠️ Note: L'analyse d'image est simulée car les modèles multimodaux ne sont pas disponibles via l'API gratuite. Le modèle analyse la description de l'image.")
    
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
                result = analyze_image_api(image, custom_prompt, selected_model, api_token)
                st.markdown("### 📊 Résultats de l'Analyse")
                st.write(result)

elif mode == "💬 Mode Chat":
    st.header("💬 Mode Chat")
    st.info("💡 Posez des questions sur l'agriculture - API directe !")
    
    # Historique des messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Afficher l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Zone de saisie alternative si chat_input ne fonctionne pas
    st.markdown("---")
    st.subheader("💭 Posez votre question")
    
    # Input utilisateur avec text_input
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
                # Préparer le contexte
                context = "Tu es un expert en agriculture. Réponds de manière utile et précise."
                full_prompt = f"{context}\n\nQuestion: {user_question}\n\nRéponse:"
                
                response = call_api(full_prompt, selected_model, api_token)
                
                # Si l'API échoue, utiliser le fallback local
                if response.startswith("❌"):
                    st.warning("⚠️ L'API Hugging Face n'est pas disponible. Utilisation du mode local.")
                    response = generate_local_response(user_question)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Recharger la page pour afficher les nouveaux messages
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
    st.info("🎯 Version simplifiée pour tester l'API")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image agricole...",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image à analyser", use_container_width=True)
        
        if st.button("🚀 Analyser avec l'API", type="primary"):
            result = analyze_image_api(image, "", selected_model, api_token)
            st.markdown("### 📊 Analyse")
            st.write(result)

# Informations sur l'API
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Informations")
st.sidebar.info(
    "**Avantages de l'API :**\n"
    "✅ Aucun téléchargement\n"
    "✅ Démarrage instantané\n"
    "✅ Modèles toujours à jour\n"
    "✅ Pas d'espace disque requis"
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🌾 AgriLens AI - API Directe<br>
        Powered by Hugging Face Inference API
    </div>
    """,
    unsafe_allow_html=True
) 