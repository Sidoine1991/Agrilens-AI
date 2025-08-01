# 🔧 DOCUMENTATION TECHNIQUE - AGRILENS AI

## 📋 **TABLE DES MATIÈRES**
1. [Architecture Générale](#architecture-générale)
2. [Structure du Code](#structure-du-code)
3. [Système de Traduction](#système-de-traduction)
4. [Gestion des Modèles](#gestion-des-modèles)
5. [Interface Utilisateur](#interface-utilisateur)
6. [Mode Mobile](#mode-mobile)
7. [API et Fonctions](#api-et-fonctions)
8. [Déploiement](#déploiement)
9. [Tests et Debugging](#tests-et-debugging)
10. [Contributions](#contributions)

---

## 🏗️ **ARCHITECTURE GÉNÉRALE**

### **Vue d'Ensemble**
```
AgriLens AI
├── Interface Utilisateur (Streamlit)
│   ├── Mode Desktop
│   └── Mode Mobile
├── Système de Traduction
├── Gestion des Modèles
│   ├── Chargement
│   ├── Persistance
│   └── Cache
├── Analyse d'Images
├── Analyse de Texte
└── Configuration
```

### **Technologies Utilisées**
- **Framework Web** : Streamlit 1.28+
- **IA/ML** : Transformers, PyTorch
- **Modèle** : Gemma 3n E4B IT (Google)
- **Traitement d'Images** : PIL (Pillow)
- **Interface** : CSS personnalisé, HTML
- **Déploiement** : Hugging Face Spaces, Local

### **Dépendances Principales**
```python
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.35.0
Pillow>=10.0.0
huggingface_hub>=0.19.0
psutil>=5.9.0
```

---

## 📁 **STRUCTURE DU CODE**

### **Organisation des Fichiers**
```
src/
├── streamlit_app_multilingual.py  # Application principale
├── README.md                      # Documentation du dossier
└── [autres versions]

docs/
├── user_manual_en.md             # Manuel utilisateur EN
├── user_manual_fr.md             # Manuel utilisateur FR
└── [documentation]

app/
├── services/                     # Services externes
│   ├── ollama_gemma_service.py
│   └── vision_caption_service.py
└── [autres services]

config/
├── config.json                   # Configuration HF Spaces
├── config_local.py              # Configuration locale
└── [autres configs]
```

### **Structure du Code Principal**
```python
# 1. Imports et Configuration
import streamlit as st
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

# 2. Configuration de la Page
st.set_page_config(...)

# 3. Système de Traduction
TRANSLATIONS = {...}
def t(key): ...

# 4. Gestion du Mode Mobile
def is_mobile(): ...
def toggle_mobile_mode(): ...

# 5. Gestion des Modèles
def load_model(): ...
def check_model_persistence(): ...

# 6. Fonctions d'Analyse
def analyze_image_multilingual(): ...
def analyze_text_multilingual(): ...

# 7. Interface Utilisateur
# - Sidebar (Configuration)
# - Onglets principaux
# - Affichage conditionnel mobile/desktop
```

---

## 🌐 **SYSTÈME DE TRADUCTION**

### **Architecture**
Le système de traduction utilise un dictionnaire simple et efficace :

```python
TRANSLATIONS = {
    "key": {
        "fr": "texte français",
        "en": "english text"
    }
}
```

### **Fonction de Traduction**
```python
def t(key):
    """
    Fonction de traduction simple pour l'interface multilingue.
    
    Args:
        key (str): Clé de traduction à rechercher
        
    Returns:
        str: Texte traduit dans la langue actuelle
    """
    if 'language' not in st.session_state:
        st.session_state.language = 'fr'
    lang = st.session_state.language
    return TRANSLATIONS.get(key, {}).get(lang, key)
```

### **Ajout de Nouvelles Traductions**
1. Ajouter la clé dans `TRANSLATIONS`
2. Fournir les traductions FR et EN
3. Utiliser `t("nouvelle_cle")` dans le code

### **Exemple d'Extension**
```python
# Ajouter une nouvelle traduction
TRANSLATIONS["new_feature"] = {
    "fr": "Nouvelle fonctionnalité",
    "en": "New feature"
}

# Utilisation
st.write(t("new_feature"))
```

---

## 🤖 **GESTION DES MODÈLES**

### **Architecture de Chargement**
```python
def load_model():
    """
    Charge le modèle Gemma 3n avec gestion d'erreurs et fallbacks.
    
    Returns:
        tuple: (model, processor) ou (None, None) en cas d'échec
    """
    try:
        # Stratégie de chargement adaptative
        return load_model_strategy(
            model_identifier="google/gemma-3n-E4B-it",
            device_map="auto",
            torch_dtype="auto"
        )
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")
        return None, None
```

### **Stratégies de Chargement**
1. **Chargement Standard** : Modèle complet avec optimisations
2. **Chargement Léger** : Pour environnements limités
3. **Chargement CPU** : Fallback sans GPU
4. **Chargement Quantifié** : 4-bit ou 8-bit pour économiser la mémoire

### **Persistance du Modèle**
```python
def force_model_persistence():
    """
    Force la persistance du modèle dans le cache de session.
    """
    st.session_state.global_model_cache['model'] = st.session_state.model
    st.session_state.global_model_cache['processor'] = st.session_state.processor
    st.session_state.global_model_cache['load_time'] = time.time()
```

### **Gestion de la Mémoire**
- **Nettoyage automatique** : `gc.collect()` après utilisation
- **Cache intelligent** : Réutilisation des modèles chargés
- **Monitoring** : Affichage de l'utilisation RAM
- **Optimisation** : Redimensionnement automatique des images

---

## 🖥️ **INTERFACE UTILISATEUR**

### **Structure de l'Interface**
```python
# Configuration de base
st.set_page_config(
    page_title="AgriLens AI - Diagnostic des Plantes",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Interface conditionnelle
if is_mobile():
    # Mode Mobile
    st.markdown('<div class="mobile-container">', unsafe_allow_html=True)
    # Interface mobile...
else:
    # Mode Desktop
    st.markdown('<div class="desktop-container">', unsafe_allow_html=True)
    # Interface desktop...
```

### **Sidebar (Configuration)**
- **Sélection de langue** : FR/EN
- **Gestion du modèle** : Chargement, statut, rechargement
- **Jeton HF** : Configuration et vérification
- **Informations système** : RAM, device, etc.

### **Onglets Principaux**
1. **📸 Analyse d'Image** : Upload et analyse d'images
2. **💬 Analyse de Texte** : Description et diagnostic textuel
3. **📖 Manuel** : Documentation utilisateur
4. **ℹ️ À propos** : Informations sur l'application

### **Gestion des États**
```python
# États de session
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'language' not in st.session_state:
    st.session_state.language = 'fr'
if 'mobile_mode' not in st.session_state:
    st.session_state.mobile_mode = False
```

---

## 📱 **MODE MOBILE**

### **Détection du Mode**
```python
def is_mobile():
    """
    Détecte si l'utilisateur est en mode mobile.
    """
    return st.session_state.get('mobile_mode', False)
```

### **Basculement de Mode**
```python
def toggle_mobile_mode():
    """
    Bascule entre le mode desktop et mobile.
    """
    if 'mobile_mode' not in st.session_state:
        st.session_state.mobile_mode = False
    st.session_state.mobile_mode = not st.session_state.mobile_mode
```

### **CSS Mobile**
```css
.mobile-container {
    max-width: 375px !important;
    margin: 0 auto !important;
    border: 2px solid #ddd !important;
    border-radius: 20px !important;
    padding: 20px !important;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
}
```

### **Composants Mobile**
- **Header mobile** : Titre avec statut offline
- **Boutons arrondis** : Interface tactile
- **Onglets stylisés** : Design mobile-friendly
- **Responsive** : Adaptation automatique

---

## 🔌 **API ET FONCTIONS**

### **Analyse d'Images**
```python
def analyze_image_multilingual(image, prompt=""):
    """
    Analyse une image de plante avec le modèle Gemma 3n.
    
    Args:
        image (PIL.Image): Image à analyser
        prompt (str): Prompt personnalisé (optionnel)
        
    Returns:
        str: Diagnostic généré par l'IA
    """
    try:
        # Préparation de l'image
        image = resize_image_if_needed(image)
        
        # Formatage du prompt
        final_prompt = f"<image>\n{prompt or 'Analyse cette image de plante...'}"
        
        # Analyse avec le modèle
        inputs = processor(text=final_prompt, images=image, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=500)
        
        # Décodage et nettoyage
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return clean_response(response, final_prompt)
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse: {e}"
```

### **Analyse de Texte**
```python
def analyze_text_multilingual(text):
    """
    Analyse une description textuelle de symptômes.
    
    Args:
        text (str): Description des symptômes
        
    Returns:
        str: Diagnostic et recommandations
    """
    try:
        # Prompt spécialisé pour l'analyse textuelle
        prompt = f"Analyse ces symptômes de plante: {text}"
        
        # Génération avec le modèle
        inputs = processor(text=prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=400)
        
        # Traitement de la réponse
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return clean_response(response, prompt)
        
    except Exception as e:
        return f"❌ Erreur lors de l'analyse de texte: {e}"
```

### **Utilitaires**
```python
def resize_image_if_needed(image, max_size=(800, 800)):
    """Redimensionne l'image si nécessaire."""
    
def afficher_ram_disponible(context=""):
    """Affiche l'utilisation de la RAM."""
    
def clean_response(response, prompt):
    """Nettoie la réponse du modèle."""
```

---

## 🚀 **DÉPLOIEMENT**

### **Déploiement Local**
```bash
# Installation des dépendances
pip install -r requirements.txt

# Lancement
streamlit run src/streamlit_app_multilingual.py --server.port 8501
```

### **Déploiement Hugging Face Spaces**
```yaml
# config.json
{
  "app_file": "src/streamlit_app_multilingual.py",
  "sdk": "streamlit"
}
```

```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/streamlit_app_multilingual.py"]
```

### **Variables d'Environnement**
```bash
# Jeton Hugging Face (recommandé)
HF_TOKEN=your_token_here

# Configuration locale
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### **Optimisations de Déploiement**
- **Cache des modèles** : Réutilisation entre sessions
- **Compression d'images** : Réduction de la bande passante
- **Gestion mémoire** : Nettoyage automatique
- **Fallbacks** : Modèles alternatifs en cas d'échec

---

## 🧪 **TESTS ET DEBUGGING**

### **Tests Unitaires**
```python
def test_translation_system():
    """Test du système de traduction."""
    assert t("title") == "AgriLens AI"
    
def test_mobile_mode():
    """Test du mode mobile."""
    assert is_mobile() == False
    toggle_mobile_mode()
    assert is_mobile() == True
```

### **Tests d'Intégration**
```python
def test_model_loading():
    """Test du chargement du modèle."""
    model, processor = load_model()
    assert model is not None
    assert processor is not None
```

### **Debugging**
```python
# Affichage des informations de debug
st.write(f"Debug: {st.session_state}")

# Monitoring des performances
import time
start_time = time.time()
# ... code à mesurer ...
st.write(f"Temps d'exécution: {time.time() - start_time:.2f}s")
```

### **Logs et Monitoring**
- **Logs Streamlit** : `streamlit run --logger.level debug`
- **Monitoring mémoire** : Affichage RAM en temps réel
- **Erreurs utilisateur** : Messages d'erreur explicites
- **Performance** : Temps de chargement et d'analyse

---

## 🤝 **CONTRIBUTIONS**

### **Guide de Contribution**
1. **Fork** le repository
2. **Créer** une branche feature
3. **Développer** avec tests
4. **Documenter** les changements
5. **Soumettre** une pull request

### **Standards de Code**
```python
# Documentation des fonctions
def ma_fonction(param1, param2):
    """
    Description courte de la fonction.
    
    Args:
        param1 (type): Description du paramètre
        param2 (type): Description du paramètre
        
    Returns:
        type: Description du retour
        
    Raises:
        ExceptionType: Description de l'exception
    """
    pass
```

### **Tests Requis**
- Tests unitaires pour nouvelles fonctions
- Tests d'intégration pour nouvelles fonctionnalités
- Tests de performance pour optimisations
- Tests de compatibilité pour nouveaux modèles

### **Documentation**
- Mettre à jour le README.md
- Documenter les nouvelles API
- Ajouter des exemples d'utilisation
- Maintenir la cohérence des traductions

---

## 📚 **RESSOURCES SUPPLÉMENTAIRES**

### **Documentation Externe**
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Gemma 3n Model Card](https://huggingface.co/google/gemma-3n-E4B-it)

### **Exemples et Tutoriels**
- [Streamlit Examples](https://docs.streamlit.io/knowledge-base/tutorials)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### **Communauté**
- [Streamlit Community](https://discuss.streamlit.io/)
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [PyTorch Forums](https://discuss.pytorch.org/)

---

## 🔮 **ROADMAP FUTURE**

### **Fonctionnalités Planifiées**
- [ ] Support pour plus de langues
- [ ] Modèles spécialisés par culture
- [ ] API REST pour intégration
- [ ] Application mobile native
- [ ] Base de données de diagnostics
- [ ] Système de recommandations avancé

### **Améliorations Techniques**
- [ ] Optimisation des performances
- [ ] Support multi-GPU
- [ ] Modèles quantifiés avancés
- [ ] Cache distribué
- [ ] Monitoring avancé

### **Expansion**
- [ ] Support pour plus de cultures
- [ ] Intégration avec capteurs IoT
- [ ] Analyse prédictive
- [ ] Collaboration avec experts agronomes

---

*Documentation technique créée par Sidoine Kolaolé YEBADOKPO - Version 2.0 - Juillet 2025*

---

## 📄 **LICENCE ET CONFORMITÉ COMPÉTITION**

### **Licence CC BY 4.0**
Ce projet est développé pour la **Google - Gemma 3n Hackathon** et est sous licence **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

**Conformité aux règles de compétition :**
- ✅ **Licence requise** : CC BY 4.0 (conforme)
- ✅ **Code open source** : Accessible et reproductible
- ✅ **Documentation complète** : Instructions détaillées
- ✅ **Utilisation Gemma 3n** : Modèle principal conforme
- ✅ **Pas de données externes coûteuses** : Utilise uniquement des ressources gratuites
- ✅ **Reproductibilité** : Code complet avec environnement

### **Obligations du gagnant (si applicable)**
- 📋 **Code source** : Entièrement disponible
- 📋 **Documentation** : Instructions de reproduction
- 📋 **Licence** : CC BY 4.0 sans restrictions commerciales
- 📋 **Attribution** : Sidoine Kolaolé YEBADOKPO 