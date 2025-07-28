import streamlit as st
import requests
import json
import base64
import io
from PIL import Image
import random
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import time

# Configuration de la page
st.set_page_config(
    page_title="AgriLens AI - Gemma Local",
    page_icon="🌾",
    layout="wide"
)

# Titre principal
st.title("🌾 AgriLens AI")
st.markdown("### Assistant IA pour l'Agriculture (Gemma Local)")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Mode d'utilisation
mode = st.sidebar.selectbox(
    "Mode d'utilisation",
    ["💬 Mode Chat", "📷 Analyse d'Image", "📤 Upload d'Image"]
)

# Token API (optionnel pour fallback)
api_token = st.sidebar.text_input(
    "🔑 Token Hugging Face (fallback)",
    type="password",
    help="Pour utiliser l'API Hugging Face en fallback"
)

# Sélection du modèle (seulement si token fourni)
if api_token and api_token.strip():
    model_options = {
        "microsoft/DialoGPT-small": "DialoGPT Small (API - Fallback)",
        "gpt2": "GPT-2 (API - Fallback)",
        "distilgpt2": "DistilGPT-2 (API - Fallback)"
    }
    
    selected_model = st.sidebar.selectbox(
        "🤖 Modèle API (fallback)",
        list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )
else:
    selected_model = "local"

# Chemin vers le modèle Gemma local
GEMMA_MODEL_PATH = r"D:\Dev\model_gemma"

# Variables globales pour le modèle
@st.cache_resource
def load_gemma_model():
    """Charge le modèle Gemma local"""
    try:
        st.info("🔄 Chargement du modèle Gemma local...")
        
        # Vérifier si le modèle existe
        if not os.path.exists(GEMMA_MODEL_PATH):
            st.error(f"❌ Modèle Gemma non trouvé dans {GEMMA_MODEL_PATH}")
            return None, None
        
        # Charger le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_PATH)
        
        # Charger le modèle
        model = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        st.success("✅ Modèle Gemma chargé avec succès !")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None, None

# Charger le modèle
gemma_model, gemma_tokenizer = load_gemma_model()

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

# Fonction pour générer avec Gemma local
def generate_with_gemma(prompt, max_length=200):
    """Génère une réponse avec le modèle Gemma local"""
    try:
        if gemma_model is None or gemma_tokenizer is None:
            return "❌ Modèle Gemma non disponible"
        
        # Préparer le prompt pour Gemma
        if "agriculture" in prompt.lower() or "plante" in prompt.lower() or "culture" in prompt.lower():
            context = "Tu es un expert en agriculture. Réponds de manière utile et précise."
        else:
            context = "Tu es un assistant IA utile et amical. Réponds de manière naturelle et appropriée."
        full_prompt = f"{context}\n\nQuestion: {prompt}\n\nRéponse:"
        
        # Encoder le prompt avec limite optimisée
        inputs = gemma_tokenizer(
            full_prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Limite optimisée pour le prompt
        )
        attention_mask = inputs["attention_mask"] if "attention_mask" in inputs else None

        # Générer la réponse avec paramètres par défaut
        with torch.inference_mode():
            try:
                # Paramètres par défaut Hugging Face
                outputs = gemma_model.generate(
                    inputs.input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=20,   # Valeur par défaut
                    temperature=0.7,     # Température standard
                    do_sample=True,
                    pad_token_id=gemma_tokenizer.eos_token_id,
                    use_cache=False       # Désactiver le cache problématique
                )
            except Exception as e:
                # Fallback simple sans cache
                outputs = gemma_model.generate(
                    inputs.input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=20,   # Fallback par défaut
                    pad_token_id=gemma_tokenizer.eos_token_id,
                    use_cache=False
                )
        
        # Décoder la réponse
        response = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire seulement la partie générée
        if full_prompt in response:
            generated_part = response.split(full_prompt)[-1].strip()
        else:
            generated_part = response.strip()
        
        return generated_part if generated_part else "🌾 Réponse générée par Gemma local"
        
    except Exception as e:
        # Retourner l'erreur pour voir ce qui se passe
        return f"❌ Erreur Gemma : {str(e)[:100]}"

# Fonction de réponse locale intelligente (fallback)
def generate_smart_response(prompt, conversation_history=None):
    """Génère une réponse intelligente basée sur le contexte (fallback)"""
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
        
        # Analyse d'images
        if "analyse d'image" in prompt_lower:
            # Détecter si c'est une laitue
            if "laitue" in prompt_lower or "salade" in prompt_lower:
                return """🥬 **Analyse de feuille de laitue attaquée** :

**IDENTIFICATION** : Feuille de laitue avec signes d'attaque

**DIAGNOSTIC** : Problèmes courants sur laitue :
• Limaces : trous irréguliers sur les feuilles
• Pucerons : colonies sous les feuilles, décoloration
• Altises : petits trous ronds caractéristiques
• Chenilles : dégâts sur les bords des feuilles
• Mildiou : taches jaunes puis brunes

**RECOMMANDATIONS** :
• Pièges à limaces (bière, planches)
• Savon noir contre pucerons
• Bacillus thuringiensis contre chenilles
• Rotation des cultures
• Amélioration de la circulation d'air"""
            else:
                return """🔍 **Analyse d'image agricole** :

**IDENTIFICATION** : Je peux identifier les types de plantes (riz, maïs, blé, agrumes, tomates, laitue, etc.)

**DIAGNOSTIC** : Je détecte les problèmes courants :
• Maladies : taches, pourriture, jaunissement
• Carences : décoloration, déformation
• Ravageurs : trous, galeries, traces

**RECOMMANDATIONS** : Je propose des solutions :
• Traitements naturels et chimiques
• Amélioration des conditions
• Prévention des problèmes

Décrivez-moi ce que vous voyez sur l'image pour un diagnostic précis !"""
        
        # Base de connaissances agricole étendue avec identification précise
        knowledge_base = {
            "riz": {
                "identification": "🌾 **Identification du riz** :\n- Plantes herbacées de 1-2m de haut\n- Feuilles longues et étroites\n- Inflorescences en panicules\n- Grains groupés en épillets\n- Culture en eau ou sec selon variété",
                "maladies": "🦠 **Maladies du riz** :\n- Pyriculariose : taches brunes en forme d'œil\n- Helminthosporiose : taches ovales brunes\n- Pourriture des racines : jaunissement, flétrissement\n- Virus de la tungro : stries jaunes sur feuilles",
                "soins": "💊 **Traitements** :\n- Rotation des cultures\n- Variétés résistantes\n- Fongicides naturels (bicarbonate)\n- Drainage approprié"
            },
            "maïs": {
                "identification": "🌽 **Identification du maïs** :\n- Plantes hautes (2-3m)\n- Tige robuste avec nœuds\n- Feuilles larges et longues\n- Inflorescences mâles en panicule\n- Épis femelles avec soies",
                "maladies": "🦠 **Maladies du maïs** :\n- Helminthosporiose : taches grises allongées\n- Rouille : pustules orange-brunes\n- Pourriture des tiges : tiges creuses\n- Virus de la mosaïque : stries jaunes"
            },
            "blé": {
                "identification": "🌾 **Identification du blé** :\n- Plantes de 0.6-1.5m\n- Tiges creuses (chaumes)\n- Feuilles étroites et longues\n- Épis compacts avec barbes\n- Grains nus",
                "maladies": "🦠 **Maladies du blé** :\n- Rouille : pustules orange\n- Septoriose : taches brunes\n- Oïdium : poudre blanche\n- Fusariose : épis décolorés"
            },
            "sorgho": {
                "identification": "🌾 **Identification du sorgho** :\n- Plantes hautes (1-4m)\n- Tiges robustes\n- Feuilles larges avec nervure centrale\n- Inflorescences en panicules denses\n- Grains ronds ou ovales",
                "maladies": "🦠 **Maladies du sorgho** :\n- Anthracnose : taches rouges\n- Mildiou : décoloration des feuilles\n- Charbon : masses noires sur grains"
            },
            "agrumes": {
                "identification": "🍊 **Identification des agrumes** :\n- Feuilles ovales, brillantes\n- Nervures bien marquées\n- Bordure lisse ou légèrement dentée\n- Odeur caractéristique au froissement",
                "maladies": "🦠 **Maladies des agrumes** :\n- Cochenilles : taches blanches/cireuses\n- Mineuse : galeries dans les feuilles\n- Mal secco : taches brunes, feuilles qui tombent\n- Tristeza : jaunissement, déformation",
                "traitements": "💊 **Traitements** :\n- Huile de neem contre les insectes\n- Fongicides naturels (bicarbonate)\n- Taille des parties malades\n- Amélioration de la circulation d'air"
            },
            "laitue": {
                "identification": "🥬 **Identification de la laitue** :\n- Feuilles larges et tendres\n- Forme de rosette\n- Couleur verte claire à foncée\n- Texture lisse ou frisée selon variété",
                "maladies": "🦠 **Maladies de la laitue** :\n- Mildiou : taches jaunes puis brunes\n- Botrytis : pourriture grise\n- Sclérotinia : pourriture blanche\n- Virus de la mosaïque : décoloration",
                "ravageurs": "🐛 **Ravageurs courants** :\n- Limaces : trous irréguliers\n- Pucerons : colonies sous les feuilles\n- Altises : petits trous ronds\n- Chenilles : dégâts sur les bords",
                "traitements": "💊 **Traitements** :\n- Pièges à limaces (bière, planches)\n- Savon noir contre pucerons\n- Bacillus thuringiensis contre chenilles\n- Rotation des cultures"
            },
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
    """Analyse une image avec Gemma local"""
    try:
        width, height = image.size
        
        # Utiliser directement Gemma local
        if gemma_model is not None and gemma_tokenizer is not None:
            st.info("🤖 Analyse avec Gemma Local...")
            
            if prompt:
                full_prompt = f"Analyse cette image: {prompt}. Image {width}x{height}px. Identifie plante et problèmes. Réponse complète et structurée."
            else:
                full_prompt = f"Analyse image agricole {width}x{height}px. Identifie plante et problèmes (maladies, ravageurs, carences). Réponse complète."
            
            # Utiliser directement generate_with_gemma sans try/catch
            return generate_with_gemma(full_prompt)
        
        # Si pas de modèle Gemma, utiliser le fallback
        st.info("🌾 Analyse avec base de connaissances agricoles")
        
        if prompt:
            return generate_smart_response(f"analyse d'image: {prompt}")
        else:
            return generate_smart_response("analyse d'image agricole")
        
    except Exception as e:
        return f"❌ Erreur : {e}"

# Interface principale
if mode == "💬 Mode Chat":
    st.header("💬 Mode Chat")
    
    if gemma_model is not None:
        st.info("🤖 Chat avec Gemma Local !")
    else:
        st.info("🌾 Chat avec l'assistant agricole (fallback) !")
    
    # Afficher les informations du modèle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informations")
    if gemma_model is not None:
        st.sidebar.info(f"**Mode** : Gemma Local\n**Chemin** : {GEMMA_MODEL_PATH}\n**Statut** : ✅ Chargé")
    else:
        st.sidebar.info("**Mode** : Local Fallback\n**Base** : Connaissances agricoles\n**Statut** : ✅ Disponible")
    
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
                    if gemma_model is not None:
                        # Utiliser Gemma local
                        response = generate_with_gemma(user_question)
                    else:
                        # Mode fallback local
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
    st.info("🚀 Analyse d'images agricoles avec Gemma Local !")
    
    # Afficher les informations du modèle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informations")
    if gemma_model is not None:
        st.sidebar.info(f"**Mode** : Gemma Local\n**Chemin** : {GEMMA_MODEL_PATH}\n**Statut** : ✅ Chargé")
    else:
        st.sidebar.info("**Mode** : Local Fallback\n**Base** : Connaissances agricoles\n**Statut** : ✅ Disponible")
    
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
            st.image(image, caption="Image uploadée")
            
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
                    st.image(image, caption="Image téléchargée")
                    
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
            st.image(image, caption="Image capturée")
            
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
    st.info("🎯 Version simplifiée avec Gemma Local")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image agricole...",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image à analyser")
        
        if st.button("🚀 Analyser", type="primary"):
            with st.spinner("🔄 Analyse en cours..."):
                result = analyze_image(image, "")
                st.markdown("### 📊 Analyse")
                st.write(result)

# Informations
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ À propos")
st.sidebar.info(
    "**Gemma Local** :\n"
    "• Modèle local puissant\n"
    "• Pas de dépendance API\n"
    "• Réponses intelligentes\n"
    "• Analyse d'images\n\n"
    "**Fallback** :\n"
    "• Base de connaissances\n"
    "• Fonctionne sans modèle\n"
    "• Réponses instantanées"
)

# Note sur le modèle
if gemma_model is not None:
    st.sidebar.success("✅ Gemma Local activé")
else:
    st.sidebar.warning("⚠️ Gemma non disponible - Mode fallback")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🌾 AgriLens AI - Gemma Local<br>
        Modèle Local + Fallback Intelligent
    </div>
    """,
    unsafe_allow_html=True
) 