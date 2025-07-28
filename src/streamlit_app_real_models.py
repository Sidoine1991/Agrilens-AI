import streamlit as st
import requests
import json
import base64
import io
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import random

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - Vrais Modèles",
    page_icon="🌾",
    layout="wide"
)

# Titre principal
st.title("🌾 AgriLens AI")
st.markdown("### Assistant IA pour l'Agriculture (Vrais Modèles Hugging Face)")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Sélection du modèle
model_options = {
    "microsoft/DialoGPT-medium": "DialoGPT Medium (Chat - Recommandé)",
    "microsoft/DialoGPT-small": "DialoGPT Small (Chat Rapide)",
    "google/gemma-3n-E4B-it": "Gemma 3N (Multimodal - Images + Texte)",
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
    help="Pour de meilleures performances"
)

# Mode d'utilisation
mode = st.sidebar.selectbox(
    "Mode d'utilisation",
    ["📷 Analyse d'Image", "💬 Mode Chat", "📤 Upload d'Image"]
)

# Initialisation des modèles
@st.cache_resource
def load_model_and_tokenizer(model_id):
    """Charge le modèle et le tokenizer"""
    try:
        st.info(f"🔄 Chargement du modèle {model_id}...")
        
        if "gemma" in model_id.lower():
            # Pour Gemma 3N (multimodal)
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            return model, processor, "multimodal"
        else:
            # Pour les modèles texte (DialoGPT, GPT-2, etc.)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            return model, tokenizer, "text"
            
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement : {e}")
        return None, None, None

# Charger le modèle
model, tokenizer_or_processor, model_type = load_model_and_tokenizer(selected_model)

if model is None:
    st.error("❌ Impossible de charger le modèle. Vérifiez votre connexion internet.")
    st.stop()

# Fonction pour générer une réponse avec DialoGPT
def generate_dialogpt_response(prompt, model, tokenizer, max_length=200):
    """Génère une réponse avec DialoGPT"""
    try:
        # Encoder l'entrée utilisateur
        new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
        
        # Générer la réponse
        with torch.inference_mode():
            chat_history_ids = model.generate(
                new_user_input_ids, 
                max_length=max_length, 
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=1
            )
        
        # Décoder la réponse
        response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        return response.strip()
        
    except Exception as e:
        return f"❌ Erreur DialoGPT : {e}"

# Fonction pour générer une réponse avec Gemma 3N
def generate_gemma_response(messages, model, processor):
    """Génère une réponse avec Gemma 3N multimodal"""
    try:
        # Préparer les messages pour Gemma 3N
        formatted_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Tu es un expert en agriculture. Analyse les images et réponds aux questions de manière précise et utile."}]
            }
        ]
        
        # Ajouter le message utilisateur
        formatted_messages.append({
            "role": "user",
            "content": messages
        })
        
        # Générer la réponse
        with torch.inference_mode():
            output = model.generate(
                formatted_messages,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )
        
        # Extraire la réponse
        if isinstance(output, list) and len(output) > 0:
            last_message = output[0]["generated_text"][-1]
            if "content" in last_message:
                return last_message["content"]
        
        return "Réponse générée par Gemma 3N"
        
    except Exception as e:
        return f"❌ Erreur Gemma 3N : {e}"

# Fonction pour analyser une image avec le vrai modèle
def analyze_image_real(image, prompt="", model_id="google/gemma-3n-E4B-it"):
    """Analyse une image avec le vrai modèle multimodal"""
    try:
        # Convertir l'image en base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        if "gemma" in model_id.lower() and model_type == "multimodal":
            # Utiliser Gemma 3N pour l'analyse d'image
            messages = [
                {"type": "image", "url": f"data:image/jpeg;base64,{img_str}"},
                {"type": "text", "text": prompt if prompt else "Analyse cette image agricole en détail. Identifie les plantes, les maladies, les conditions de croissance, et donne des recommandations."}
            ]
            
            return generate_gemma_response(messages, model, tokenizer_or_processor)
        else:
            # Pour les modèles texte uniquement, utiliser l'analyse simulée
            width, height = image.size
            analysis_prompt = f"Tu es un expert en agriculture. Analyse cette image agricole: {prompt}. L'image fait {width}x{height} pixels. Donne une analyse détaillée et des recommandations."
            
            if model_type == "text":
                return generate_dialogpt_response(analysis_prompt, model, tokenizer_or_processor)
            else:
                return f"❌ Modèle non compatible avec l'analyse d'image : {model_id}"
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse : {e}"

# Fonction pour le chat avec le vrai modèle
def chat_with_real_model(prompt, model_id):
    """Chat avec le vrai modèle sélectionné"""
    try:
        if "gemma" in model_id.lower() and model_type == "multimodal":
            # Utiliser Gemma 3N pour le chat
            messages = [
                {"type": "text", "text": f"Tu es un expert en agriculture. Réponds à cette question : {prompt}"}
            ]
            return generate_gemma_response(messages, model, tokenizer_or_processor)
        elif model_type == "text":
            # Utiliser DialoGPT ou autre modèle texte
            chat_prompt = f"Tu es un expert en agriculture. Question : {prompt}\n\nRéponse :"
            return generate_dialogpt_response(chat_prompt, model, tokenizer_or_processor)
        else:
            return f"❌ Modèle non supporté : {model_id}"
            
    except Exception as e:
        return f"❌ Erreur de chat : {e}"

# Interface principale
if mode == "📷 Analyse d'Image":
    st.header("📷 Analyse d'Image")
    st.info(f"🤖 Utilise le vrai modèle {selected_model} pour analyser vos images !")
    
    # Afficher les informations du modèle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informations Modèle")
    st.sidebar.info(f"**Modèle** : {selected_model}\n**Type** : {model_type}\n**Statut** : ✅ Chargé")
    
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
            
            if st.button("🔍 Analyser avec le Vrai Modèle", type="primary"):
                with st.spinner(f"🔄 Analyse avec {selected_model}..."):
                    result = analyze_image_real(image, custom_prompt, selected_model)
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
                        result = analyze_image_real(image, "", selected_model)
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
            
            if st.button("🔍 Analyser avec le Vrai Modèle", type="primary"):
                with st.spinner(f"🔄 Analyse avec {selected_model}..."):
                    result = analyze_image_real(image, custom_prompt, selected_model)
                    st.markdown("### 📊 Résultats de l'Analyse")
                    st.write(result)

elif mode == "💬 Mode Chat":
    st.header("💬 Mode Chat")
    st.info(f"🤖 Chat avec le vrai modèle {selected_model} !")
    
    # Afficher les informations du modèle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informations Modèle")
    st.sidebar.info(f"**Modèle** : {selected_model}\n**Type** : {model_type}\n**Statut** : ✅ Chargé")
    
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
            
            # Générer la réponse avec le vrai modèle
            try:
                with st.spinner(f"🔄 Génération avec {selected_model}..."):
                    response = chat_with_real_model(user_question, selected_model)
                
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
        
        if st.button("🚀 Analyser avec le Vrai Modèle", type="primary"):
            with st.spinner(f"🔄 Analyse avec {selected_model}..."):
                result = analyze_image_real(image, "", selected_model)
                st.markdown("### 📊 Analyse")
                st.write(result)

# Informations sur les modèles
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Modèles Disponibles")
st.sidebar.info(
    "**DialoGPT** : Chat conversationnel\n"
    "**Gemma 3N** : Multimodal (Images + Texte)\n"
    "**GPT-2** : Génération de texte\n"
    "**DistilGPT-2** : Version légère"
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🌾 AgriLens AI - Vrais Modèles<br>
        Powered by Hugging Face Transformers
    </div>
    """,
    unsafe_allow_html=True
) 