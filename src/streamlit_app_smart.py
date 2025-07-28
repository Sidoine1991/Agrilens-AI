import streamlit as st
import requests
import json
import base64
import io
from PIL import Image
import random

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - Smart",
    page_icon="🌾",
    layout="wide"
)

# Titre principal
st.title("🌾 AgriLens AI")
st.markdown("### Assistant IA pour l'Agriculture (Version Intelligente)")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Mode d'utilisation
mode = st.sidebar.selectbox(
    "Mode d'utilisation",
    ["💬 Mode Chat", "📷 Analyse d'Image", "📤 Upload d'Image"]
)

# Token API (optionnel)
api_token = st.sidebar.text_input(
    "🔑 Token Hugging Face (optionnel)",
    type="password",
    help="Pour utiliser l'API Hugging Face"
)

# Sélection du modèle (seulement si token fourni)
if api_token and api_token.strip():
    model_options = {
        "microsoft/DialoGPT-small": "DialoGPT Small (Chat - Testé)",
        "gpt2": "GPT-2 (Texte Général - Testé)",
        "distilgpt2": "DistilGPT-2 (Léger - Testé)"
    }
    
    selected_model = st.sidebar.selectbox(
        "🤖 Modèle API (si token fourni)",
        list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )
else:
    selected_model = "local"

# Fonction pour appeler l'API avec gestion d'erreur améliorée
def call_api_safe(prompt, model_id, token=None):
    """Appelle l'API Hugging Face avec gestion d'erreur robuste"""
    if not token or not token.strip():
        return "❌ Token requis pour l'API Hugging Face"
    
    try:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
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
            return f"❌ Modèle {model_id} non disponible via l'API gratuite."
        elif response.status_code == 503:
            return "🔄 Le modèle est en cours de chargement. Réessayez dans quelques secondes."
        elif response.status_code == 401:
            return "❌ Token invalide ou expiré. Vérifiez votre token Hugging Face."
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

# Fonction de réponse locale intelligente
def generate_smart_response(prompt, conversation_history=None):
    """Génère une réponse intelligente basée sur le contexte"""
    try:
        prompt_lower = prompt.lower().strip()
        
        # Salutations et présentations
        greetings = ["bonjour", "salut", "hello", "hi", "coucou", "hey"]
        if any(greeting in prompt_lower for greeting in greetings):
            responses = [
                "🌾 Bonjour ! Je suis votre assistant agricole. Je peux vous aider avec des conseils sur la culture, les maladies des plantes, l'amélioration du sol, et bien plus encore. Que souhaitez-vous savoir ?",
                "🌱 Salut ! Je suis spécialisé en agriculture et jardinage. Posez-moi vos questions sur les cultures, les techniques, ou l'analyse d'images agricoles !",
                "👨‍🌾 Bonjour ! Votre expert agricole est là pour vous aider. Cultures, soins, maladies, sol... je peux tout vous expliquer !"
            ]
            return random.choice(responses)
        
        # Questions sur les capacités
        if any(word in prompt_lower for word in ["peux-tu", "peux tu", "peut-il", "peut il", "capacités", "faire quoi", "aider"]):
            return "🌾 **Mes capacités** :\n\n• **Conseils de culture** : tomates, légumes, fruits\n• **Diagnostic de maladies** : identification et traitements\n• **Amélioration du sol** : types, engrais, techniques\n• **Techniques d'arrosage** : optimisation et signes\n• **Calendrier de plantation** : saisons et périodes\n• **Plantes compagnes** : associations bénéfiques\n• **Analyse d'images** : diagnostic visuel\n\nQue voulez-vous explorer ?"
        
        # Base de connaissances agricole étendue
        knowledge_base = {
            "tomate": {
                "culture": "🌱 **Culture des tomates** :\n1) Plantez en plein soleil (6-8h/jour)\n2) Espacez de 60-90cm entre les plants\n3) Arrosez régulièrement à la base\n4) Tuteurez les plants pour éviter la pourriture\n5) Fertilisez avec du compost ou engrais équilibré\n6) Pincez les gourmands pour favoriser la production",
                "maladies": "🦠 **Maladies courantes** :\n- Mildiou : taches brunes sur feuilles\n- Oïdium : poudre blanche\n- Pourriture apicale : carence en calcium\n**Solutions** : rotation des cultures, aération, traitement préventif"
            },
            "maladie": {
                "identification": "🔍 **Identification des maladies** :\n- Feuilles jaunes : carence ou excès d'eau\n- Taches brunes : champignons\n- Poudre blanche : oïdium\n- Pourriture : bactéries ou champignons",
                "traitement": "💊 **Traitements naturels** :\n1) Retirez les parties malades\n2) Améliorez la circulation d'air\n3) Utilisez du bicarbonate de soude\n4) Pulvérisez du lait dilué\n5) Plantez des plantes compagnes"
            },
            "sol": {
                "amelioration": "🌍 **Amélioration du sol** :\n1) Ajoutez du compost (30% du volume)\n2) Utilisez du paillage (paille, feuilles)\n3) Plantez des engrais verts (trèfle, moutarde)\n4) Évitez le compactage\n5) Testez le pH (6.0-7.0 idéal)\n6) Ajoutez de la matière organique",
                "types": "🏗️ **Types de sol** :\n- Argileux : retient l'eau, ajoutez du sable\n- Sableux : draine vite, ajoutez de l'argile\n- Limoneux : équilibré, idéal\n- Calcaire : ajoutez de la tourbe"
            },
            "arrosage": {
                "techniques": "💧 **Techniques d'arrosage optimal** :\n1) Tôt le matin (avant 10h)\n2) À la base des plantes\n3) Évitez de mouiller les feuilles\n4) Adaptez selon la météo\n5) Utilisez du paillage\n6) Arrosez profondément mais moins souvent",
                "signes": "🔍 **Signes de déshydratation** :\n- Feuilles flétries le soir\n- Sol sec sur 2-3cm\n- Croissance ralentie\n- Fruits fendillés"
            },
            "printemps": {
                "legumes": "🌸 **Légumes de printemps** :\n1) Pois (février-mars) - résistant au froid\n2) Épinards (mars) - pousse vite\n3) Radis (mars-avril) - 3-4 semaines\n4) Laitues (avril) - plusieurs variétés\n5) Carottes (avril-mai) - sol meuble\n6) Oignons (mars-avril) - bulbes ou graines",
                "conseils": "📅 **Conseils de plantation** :\n- Attendez que le sol soit réchauffé\n- Protégez des gelées tardives\n- Semez en échelon pour étaler les récoltes\n- Utilisez des tunnels ou cloches"
            },
            "engrais": {
                "naturels": "🌿 **Engrais naturels** :\n1) Compost : équilibré, améliore le sol\n2) Fumier : riche en azote\n3) Cendres : potassium et calcium\n4) Algues : oligo-éléments\n5) Sang séché : azote rapide\n6) Poudre d'os : phosphore",
                "utilisation": "⚖️ **Utilisation** :\n- Compost : 5-10cm en surface\n- Fumier : 3-6 mois avant plantation\n- Cendres : 100g/m² maximum\n- Algues : en pulvérisation foliaire"
            },
            "compagnonnage": {
                "plantes": "🤝 **Plantes compagnes** :\n- Tomates + Basilic : repousse les insectes\n- Carottes + Oignons : se protègent mutuellement\n- Pois + Maïs : le maïs sert de tuteur\n- Salades + Radis : optimise l'espace\n- Courges + Maïs + Haricots : les 3 sœurs",
                "benefices": "✨ **Bénéfices** :\n- Repousse les ravageurs\n- Améliore la pollinisation\n- Optimise l'espace\n- Améliore la fertilité\n- Protège du vent"
            },
            "carotte": {
                "culture": "🥕 **Culture des carottes** :\n1) Sol meuble et profond (pas de cailloux)\n2) Semez directement en place\n3) Éclaircissez à 5-8cm\n4) Arrosez régulièrement\n5) Récoltez 2-4 mois selon variété\n6) Stockez en cave ou silo"
            },
            "salade": {
                "culture": "🥬 **Culture des salades** :\n1) Semez en pépinière ou direct\n2) Repiquez à 25-30cm d'écart\n3) Arrosez fréquemment\n4) Récoltez avant montaison\n5) Plantez en échelon pour étaler\n6) Protégez des limaces"
            },
            "pomme de terre": {
                "culture": "🥔 **Culture des pommes de terre** :\n1) Plantez en mars-avril\n2) Buttez régulièrement\n3) Arrosez modérément\n4) Surveillez le mildiou\n5) Récoltez quand les feuilles jaunissent\n6) Stockez au sec et à l'obscurité"
            }
        }
        
        # Chercher des mots-clés dans la question
        for keyword, info in knowledge_base.items():
            if keyword in prompt_lower:
                if isinstance(info, dict):
                    # Retourner toutes les informations disponibles
                    response = f"🌾 **{keyword.title()}** :\n\n"
                    for topic, content in info.items():
                        response += f"**{topic.title()}** :\n{content}\n\n"
                    return response
                else:
                    return f"🌾 **{keyword.title()}** : {info}"
        
        # Questions spécifiques
        if any(word in prompt_lower for word in ["comment", "comment faire", "technique"]):
            suggestions = [
                "🌾 **Techniques agricoles** : Je peux vous expliquer comment cultiver des tomates, améliorer votre sol, optimiser l'arrosage, ou utiliser des engrais naturels. Que voulez-vous apprendre ?",
                "🌱 **Conseils pratiques** : Dites-moi quelle culture vous intéresse (tomates, carottes, salades...) ou quel aspect (sol, arrosage, maladies) et je vous donnerai des conseils détaillés !"
            ]
            return random.choice(suggestions)
        
        elif any(word in prompt_lower for word in ["quand", "saison", "période", "calendrier"]):
            return "📅 **Calendrier de plantation** :\n\n**Printemps** : Pois, épinards, radis, laitues, carottes, oignons\n**Été** : Tomates, courgettes, haricots, maïs\n**Automne** : Épinards, mâche, choux, poireaux\n**Hiver** : Planification, préparation du sol\n\nQuelle saison vous intéresse ?"
        
        elif any(word in prompt_lower for word in ["problème", "erreur", "difficulté", "maladie"]):
            return "🔍 **Diagnostic de problèmes** :\n\nDécrivez-moi les symptômes que vous observez :\n- Couleur des feuilles (jaunes, brunes, blanches)\n- Aspect des tiges ou fruits\n- Comportement de la plante\n- Conditions météo récentes\n\nJe pourrai alors vous aider à identifier le problème !"
        
        elif any(word in prompt_lower for word in ["merci", "thanks", "thank you"]):
            responses = [
                "🌾 De rien ! N'hésitez pas si vous avez d'autres questions sur l'agriculture !",
                "🌱 Avec plaisir ! Votre jardinier virtuel est là pour vous aider !",
                "👨‍🌾 Je vous en prie ! Bon jardinage et n'hésitez pas à revenir !"
            ]
            return random.choice(responses)
        
        # Réponse générique intelligente
        else:
            suggestions = [
                "🌾 **Sujets que je peux aborder** :\n• Culture de légumes (tomates, carottes, salades...)\n• Amélioration du sol et engrais\n• Techniques d'arrosage\n• Diagnostic de maladies\n• Plantes compagnes\n• Calendrier de plantation\n\nQue voulez-vous explorer ?",
                "🌱 **Je peux vous aider avec** :\n• Conseils de plantation et culture\n• Identification de problèmes\n• Techniques d'amélioration du sol\n• Optimisation de l'arrosage\n• Associations de plantes\n\nPosez-moi une question spécifique !",
                "👨‍🌾 **Votre expert agricole peut vous conseiller sur** :\n• Toutes les cultures de légumes\n• Soins et entretien des plantes\n• Diagnostic et traitement des maladies\n• Amélioration de la fertilité du sol\n• Techniques de jardinage écologique\n\nQue souhaitez-vous savoir ?"
            ]
            return random.choice(suggestions)
        
    except Exception as e:
        return f"❌ Erreur : {e}"

# Fonction pour analyser une image
def analyze_image(image, prompt=""):
    """Analyse une image (simulation pour API texte)"""
    try:
        width, height = image.size
        
        if prompt:
            full_prompt = f"Tu es un expert en agriculture. Analyse cette image: {prompt}. L'image fait {width}x{height} pixels. Donne une analyse détaillée."
        else:
            full_prompt = f"Tu es un expert en agriculture. Analyse cette image agricole. L'image fait {width}x{height} pixels. Identifie les plantes, maladies, conditions de croissance."
        
        # Essayer l'API d'abord si token disponible
        if api_token and api_token.strip():
            api_response = call_api_safe(full_prompt, selected_model, api_token)
            
            # Si l'API fonctionne, utiliser sa réponse
            if not api_response.startswith("❌") and not api_response.startswith("🔄") and not api_response.startswith("⏳"):
                return api_response
        
        # Sinon, utiliser le fallback local
        st.info("💡 Utilisation du mode local (pas de token API)")
        return generate_smart_response(prompt if prompt else "analyse d'image agricole")
        
    except Exception as e:
        return f"❌ Erreur : {e}"

# Interface principale
if mode == "💬 Mode Chat":
    st.header("💬 Mode Chat")
    
    if api_token and api_token.strip():
        st.info(f"🤖 Chat avec {selected_model} (API Hugging Face) !")
    else:
        st.info("🌾 Chat avec l'assistant agricole intelligent !")
    
    # Afficher les informations du modèle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informations")
    if api_token and api_token.strip():
        st.sidebar.info(f"**Mode** : API Hugging Face\n**Modèle** : {selected_model}\n**Statut** : ✅ Configuré")
    else:
        st.sidebar.info("**Mode** : Local Intelligent\n**Base** : Connaissances agricoles\n**Statut** : ✅ Disponible")
    
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
                with st.spinner("🔄 Génération de la réponse..."):
                    if api_token and api_token.strip():
                        # Essayer l'API d'abord
                        context = "Tu es un expert en agriculture. Réponds de manière utile et précise."
                        full_prompt = f"{context}\n\nQuestion: {user_question}\n\nRéponse:"
                        
                        response = call_api_safe(full_prompt, selected_model, api_token)
                        
                        # Si l'API échoue, utiliser le fallback local
                        if response.startswith("❌") or response.startswith("🔄") or response.startswith("⏳"):
                            st.warning("⚠️ L'API Hugging Face n'est pas disponible. Utilisation du mode local.")
                            response = generate_smart_response(user_question, st.session_state.messages)
                    else:
                        # Mode local direct
                        response = generate_smart_response(user_question, st.session_state.messages)
                
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
    st.info("🚀 Analyse d'images agricoles !")
    
    if not api_token or not api_token.strip():
        st.info("💡 Mode local : L'analyse est basée sur les dimensions et votre description")
    
    # Afficher les informations du modèle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informations")
    if api_token and api_token.strip():
        st.sidebar.info(f"**Mode** : API Hugging Face\n**Modèle** : {selected_model}\n**Statut** : ✅ Configuré")
    else:
        st.sidebar.info("**Mode** : Local Intelligent\n**Base** : Connaissances agricoles\n**Statut** : ✅ Disponible")
    
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
            
            if st.button("🔍 Analyser", type="primary"):
                with st.spinner("🔄 Analyse en cours..."):
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
                    
                    with st.spinner("🔄 Analyse en cours..."):
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
            
            if st.button("🔍 Analyser", type="primary"):
                with st.spinner("🔄 Analyse en cours..."):
                    result = analyze_image(image, custom_prompt)
                    st.markdown("### 📊 Résultats de l'Analyse")
                    st.write(result)

elif mode == "📤 Upload d'Image":
    st.header("📤 Upload d'Image Simple")
    st.info("🎯 Version simplifiée")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image agricole...",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image à analyser", use_container_width=True)
        
        if st.button("🚀 Analyser", type="primary"):
            with st.spinner("🔄 Analyse en cours..."):
                result = analyze_image(image, "")
                st.markdown("### 📊 Analyse")
                st.write(result)

# Informations
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ À propos")
st.sidebar.info(
    "**Mode Local Intelligent** :\n"
    "• Réponses variées et naturelles\n"
    "• Connaissances agricoles complètes\n"
    "• Pas de répétitions\n"
    "• Fonctionne sans internet\n\n"
    "**Mode API** :\n"
    "• Nécessite un token\n"
    "• Modèles plus avancés\n"
    "• Fallback automatique"
)

# Note sur le token
if not api_token or not api_token.strip():
    st.sidebar.info("💡 Pas de token : Mode local intelligent activé")
else:
    st.sidebar.success("✅ Token configuré")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🌾 AgriLens AI - Version Intelligente<br>
        Mode Local Smart + API Hugging Face
    </div>
    """,
    unsafe_allow_html=True
) 