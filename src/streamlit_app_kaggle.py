import streamlit as st
import os
import io
from PIL import Image
import requests
import torch
import google.generativeai as genai
import gc
import time
import sys
import psutil
import subprocess

# Configuration Kaggle
KAGGLE_USERNAME = st.secrets.get("kaggle_username", "")
KAGGLE_KEY = st.secrets.get("kaggle_key", "")

def setup_kaggle():
    """Configure l'accès Kaggle"""
    try:
        # Créer le dossier .kaggle s'il n'existe pas
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        
        # Créer le fichier kaggle.json
        kaggle_config = {
            "username": KAGGLE_USERNAME,
            "key": KAGGLE_KEY
        }
        
        import json
        with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
            json.dump(kaggle_config, f)
        
        # Définir les permissions (important sur Linux/Mac)
        os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
        
        st.success("✅ Configuration Kaggle réussie")
        return True
    except Exception as e:
        st.error(f"❌ Erreur configuration Kaggle: {e}")
        return False

def download_model_from_kaggle(dataset_name, local_path):
    """Télécharge un modèle depuis Kaggle"""
    try:
        st.info(f"📥 Téléchargement du modèle {dataset_name} depuis Kaggle...")
        
        # Installer kaggle si nécessaire
        try:
            import kaggle
        except ImportError:
            st.info("📦 Installation de l'API Kaggle...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            import kaggle
        
        # Télécharger le dataset
        kaggle.api.dataset_download_files(dataset_name, path=local_path, unzip=True)
        st.success(f"✅ Modèle téléchargé avec succès dans {local_path}")
        return True
    except Exception as e:
        st.error(f"❌ Erreur téléchargement Kaggle: {e}")
        return False

def load_model_with_kaggle_fallback():
    """Charge le modèle avec fallback vers Kaggle"""
    try:
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        
        # Essayer d'abord Hugging Face
        st.info("🔄 Tentative de chargement depuis Hugging Face...")
        
        model_id = "google/gemma-3n-E4B-it"
        
        try:
            # Charger le processeur depuis Hugging Face
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                timeout=30  # Timeout de 30 secondes
            )
            
            # Charger le modèle
            model = Gemma3nForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                timeout=60  # Timeout de 60 secondes
            )
            
            st.success("✅ Modèle chargé depuis Hugging Face")
            return model, processor
            
        except Exception as hf_error:
            st.warning(f"⚠️ Échec Hugging Face: {hf_error}")
            
            # Essayer Kaggle comme fallback
            st.info("🔄 Tentative de chargement depuis Kaggle...")
            
            if setup_kaggle():
                # Télécharger le modèle depuis Kaggle
                kaggle_dataset = "google/gemma-3n-e4b-it"  # Exemple de dataset Kaggle
                local_path = "./models/kaggle_gemma"
                
                if download_model_from_kaggle(kaggle_dataset, local_path):
                    # Charger le modèle local téléchargé
                    processor = AutoProcessor.from_pretrained(
                        local_path,
                        trust_remote_code=True
                    )
                    
                    model = Gemma3nForConditionalGeneration.from_pretrained(
                        local_path,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    
                    st.success("✅ Modèle chargé depuis Kaggle")
                    return model, processor
                else:
                    st.error("❌ Échec du téléchargement depuis Kaggle")
            else:
                st.error("❌ Impossible de configurer Kaggle")
            
            # Fallback vers le modèle local existant
            st.info("🔄 Tentative de chargement depuis le modèle local...")
            local_model_path = "D:/Dev/model_gemma"
            
            if os.path.exists(local_model_path):
                processor = AutoProcessor.from_pretrained(
                    local_model_path,
                    trust_remote_code=True
                )
                
                model = Gemma3nForConditionalGeneration.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                st.success("✅ Modèle chargé depuis le dossier local")
                return model, processor
            else:
                st.error("❌ Aucun modèle local trouvé")
                return None, None
                
    except Exception as e:
        st.error(f"❌ Erreur générale de chargement: {e}")
        return None, None

# Interface utilisateur
st.set_page_config(
    page_title="AgriLens AI - Kaggle Integration",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 AgriLens AI - Diagnostic des Maladies de Plantes")
st.markdown("### Version avec intégration Kaggle")

# Configuration Kaggle
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Champs pour les credentials Kaggle
    st.subheader("📊 Configuration Kaggle")
    kaggle_username = st.text_input("Nom d'utilisateur Kaggle", value=KAGGLE_USERNAME)
    kaggle_key = st.text_input("Clé API Kaggle", value=KAGGLE_KEY, type="password")
    
    if st.button("🔗 Connecter Kaggle"):
        if kaggle_username and kaggle_key:
            # Mettre à jour les secrets
            st.session_state.kaggle_username = kaggle_username
            st.session_state.kaggle_key = kaggle_key
            
            if setup_kaggle():
                st.success("✅ Connexion Kaggle réussie!")
        else:
            st.error("❌ Veuillez fournir les credentials Kaggle")

# Chargement du modèle
if 'model' not in st.session_state or st.session_state.model is None:
    st.info("🔄 Chargement du modèle...")
    model, processor = load_model_with_kaggle_fallback()
    
    if model is not None and processor is not None:
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.model_loaded = True
        st.success("✅ Modèle chargé avec succès!")
    else:
        st.error("❌ Impossible de charger le modèle")
        st.stop()

# Interface principale
st.header("📸 Analyse d'Image")

uploaded_file = st.file_uploader(
    "Choisissez une image de plante à analyser",
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)
    
    if st.button("🔍 Analyser l'image"):
        with st.spinner("Analyse en cours..."):
            # Analyse avec le modèle chargé
            try:
                # Préparer l'image
                inputs = st.session_state.processor(
                    image,
                    return_tensors="pt"
                )
                
                # Générer la réponse
                with torch.no_grad():
                    outputs = st.session_state.model.generate(
                        **inputs,
                        max_length=512,
                        num_return_sequences=1,
                        temperature=0.7
                    )
                
                # Décoder la réponse
                response = st.session_state.processor.decode(outputs[0], skip_special_tokens=True)
                
                st.success("✅ Analyse terminée!")
                st.write("### 📋 Résultats de l'analyse:")
                st.write(response)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {e}")

# Informations sur les sources de modèles
with st.expander("ℹ️ Sources de modèles disponibles"):
    st.markdown("""
    ### 📊 Sources de modèles supportées:
    
    1. **🤗 Hugging Face** (par défaut)
       - Modèle: `google/gemma-3n-E4B-it`
       - Avantages: Rapide, toujours à jour
       - Inconvénients: Nécessite une connexion internet stable
    
    2. **🏆 Kaggle** (fallback)
       - Modèle: Datasets Kaggle
       - Avantages: Alternative fiable, modèles optimisés
       - Inconvénients: Nécessite un compte Kaggle
    
    3. **💾 Modèle Local** (fallback)
       - Chemin: `D:/Dev/model_gemma`
       - Avantages: Fonctionne hors ligne
       - Inconvénients: Nécessite un téléchargement préalable
    
    ### 🔧 Configuration recommandée:
    - Créez un compte Kaggle gratuit
    - Générez une clé API dans les paramètres Kaggle
    - Configurez les credentials dans l'interface
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🌱 AgriLens AI - Version avec intégration Kaggle<br>
    Développé pour la compétition Kaggle sur le diagnostic des maladies de plantes
</div>
""", unsafe_allow_html=True) 