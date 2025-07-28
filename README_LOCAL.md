# 🌱 AgriLens AI - Version Locale

**Application de diagnostic des maladies de plantes avec IA - Version Locale Complète**

## 🚀 Lancement Rapide

### Option 1 : Fichier Batch (Windows)
```bash
# Double-cliquez sur le fichier
lancer_agrilens_local.bat
```

### Option 2 : Script Python
```bash
python run_local.py
```

### Option 3 : Commande Directe
```bash
# Activer l'environnement virtuel
venv\Scripts\activate

# Lancer l'application
streamlit run src/streamlit_app_multilingual.py --server.port=8501
```

## 📋 Fonctionnalités Complètes

### 🔍 **Analyse d'Image Avancée**
- **📁 Upload d'image** : Téléchargez des photos depuis votre appareil
- **📷 Capture Webcam** : Prenez des photos directement avec votre caméra
- **🔬 Diagnostic Précis** : Analyse par Gemini AI avec identification de maladie
- **📊 Résultats Structurés** : Diagnostic, symptômes, traitement, prévention, urgence

### 💬 **Analyse de Texte**
- **📝 Description de symptômes** : Décrivez les problèmes de vos plantes
- **🧠 Analyse par Gemma 2B** : Diagnostic basé sur le texte
- **💡 Recommandations** : Conseils personnalisés

### 🌐 **Support Multilingue**
- **🇫🇷 Français** : Interface complète en français
- **🇬🇧 Anglais** : Interface complète en anglais
- **🔄 Changement dynamique** : Basculez entre les langues

### 📱 **Interface Mobile**
- **📱 Responsive Design** : Optimisé pour smartphones et tablettes
- **👆 Interface tactile** : Boutons et contrôles adaptés au touch
- **📐 Layout adaptatif** : S'adapte à toutes les tailles d'écran

### ⚙️ **Configuration Avancée**
- **🔑 API Keys intégrées** : Google Gemini et Hugging Face configurés
- **📦 Gestion des dépendances** : Vérification automatique
- **🔄 Variables d'environnement** : Configuration automatique

## 🛠️ Installation

### Prérequis
- Python 3.8+
- Git
- Connexion Internet (pour télécharger les modèles)

### Étapes d'installation

1. **Cloner le projet**
```bash
git clone <votre-repo>
cd AgriLensAI
```

2. **Créer l'environnement virtuel**
```bash
python -m venv venv
```

3. **Activer l'environnement**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

5. **Lancer l'application**
```bash
# Option simple
lancer_agrilens_local.bat

# Ou manuellement
streamlit run src/streamlit_app_multilingual.py
```

## 📖 Guide d'Utilisation

### 🔍 **Analyse d'Image**

1. **Choisissez votre méthode :**
   - **📁 Upload** : Cliquez sur "Choisir une image"
   - **📷 Webcam** : Cliquez sur "Capture par webcam"

2. **Prenez votre photo :**
   - Assurez-vous que la plante est bien éclairée
   - Photographiez les zones malades de près
   - Incluez plusieurs angles si possible

3. **Posez une question (optionnel) :**
   - "Quelle est cette maladie ?"
   - "Que faire pour la traiter ?"
   - "Est-ce urgent ?"

4. **Obtenez votre diagnostic :**
   - **Diagnostic Précis** : Nom scientifique de la maladie
   - **Symptômes Détaillés** : Description complète
   - **Traitement Recommandé** : Actions spécifiques
   - **Actions Préventives** : Mesures préventives
   - **Niveau d'Urgence** : Priorité du traitement

### 💬 **Analyse de Texte**

1. **Chargez le modèle** dans les réglages
2. **Décrivez les symptômes** observés
3. **Obtenez des conseils** personnalisés

### 🌐 **Changement de Langue**

1. **Ouvrez les réglages** (sidebar)
2. **Sélectionnez la langue** souhaitée
3. **L'interface se met à jour** automatiquement

## 🔧 Configuration

### Variables d'Environnement

Les clés API sont déjà configurées dans le code :

```python
GOOGLE_API_KEY = "AIzaSyC4a4z20p7EKq1Fk5_AX8eB_1yBo1HqYvA"
HF_TOKEN = "hf_gUGRsgWffLNZVuzYLsmTdPwESIyrbryZW"
```

### Modèles Utilisés

- **Gemini 1.5 Flash** : Analyse d'images avancée
- **Gemma 2B** : Analyse de texte et diagnostic

## 🚨 Dépannage

### Erreur "Module not found"
```bash
pip install -r requirements.txt
```

### Erreur "Model not loaded"
- Vérifiez votre connexion Internet
- Cliquez sur "Charger le modèle" dans les réglages

### Erreur "Gemini API not configured"
- Les clés sont déjà configurées dans le code
- Vérifiez votre connexion Internet

### Application ne se lance pas
```bash
# Vérifiez Python
python --version

# Réinstallez les dépendances
pip install --upgrade -r requirements.txt
```

## 📱 Compatibilité

- **Windows** : ✅ Testé et fonctionnel
- **macOS** : ✅ Compatible
- **Linux** : ✅ Compatible
- **Smartphones** : ✅ Interface responsive
- **Tablettes** : ✅ Optimisé tactile

## 🔬 Technologies

- **Streamlit** : Interface web
- **Gemini AI** : Analyse d'images avancée
- **Gemma 2B** : Analyse de texte
- **PyTorch** : Framework d'IA
- **PIL** : Traitement d'images
- **Transformers** : Modèles Hugging Face

## 👨‍💻 Créateur

**Sidoine Kolaolé YEBADOKPO**
- 📍 Bohicon, République du Bénin
- 📞 +229 01 96 91 13 46
- 📧 syebadokpo@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/sidoineko)
- 🎯 [Portfolio Hugging Face](https://huggingface.co/Sidoineko/portfolio)

## 🏆 Version Compétition

Cette version locale inclut toutes les fonctionnalités de la version Hugging Face Spaces, développée spécifiquement pour la compétition Kaggle.

## 📄 Licence

MIT License - Libre d'utilisation et de modification

## ⚠️ Avertissement

Cette application est un outil d'aide au diagnostic. Pour des cas critiques, consultez toujours un expert agricole local.

---

**🌱 AgriLens AI - Diagnostic intelligent des plantes avec IA** 