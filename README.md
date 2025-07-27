---
title: AgriLens AI
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
sdk_version: "1.0.0"
app_file: src/streamlit_app_local_models.py
pinned: false
---

# AgriLens AI 🌱

Application de diagnostic des maladies des plantes utilisant le modèle Gemma 3n multimodal de Google en mode local.

## ✨ Fonctionnalités principales
- **Analyse d'images de plantes** avec IA locale
- **Diagnostic automatique** des maladies et problèmes
- **Recommandations pratiques** avec section "Actions urgentes"
- **Interface mobile** optimisée pour les agriculteurs
- **Modèle multimodal** : analyse image + texte simultanément
- **Mode 100% offline** - aucune connexion Internet requise

## 🚀 Comment utiliser
1. **Chargez le modèle** : Cliquez sur "Charger le modèle Gemma 3n multimodal" dans la sidebar
2. **Téléchargez une photo** de plante malade
3. **Posez une question** spécifique (optionnel)
4. **Cliquez sur "Analyser avec l'IA Locale"**
5. **Consultez le diagnostic** et les recommandations

## 💻 Configuration requise
- **Hardware** : CPU (GPU optionnel)
- **Espace disque** : 20GB+ (pour le modèle Gemma 3n)
- **Mémoire RAM** : 8GB minimum recommandé
- **Système** : Windows 10/11, Linux, macOS

## ⚠️ Avertissement
Les résultats fournis sont à titre indicatif uniquement. Pour un diagnostic médical ou agricole professionnel, veuillez consulter un expert qualifié.

## 🔧 Développement
- **Framework** : Streamlit
- **Modèle** : Gemma 3n multimodal (local)
- **Génération** : Paramètres optimisés (temperature=0.7, top_p=0.9)
- **Interface** : Responsive design mobile-first
- **Dernière mise à jour** : Juillet 2025

---

## 🇫🇷 Installation rapide
1. **Téléchargez ou clonez ce dépôt**
2. **Placez le modèle Gemma 3n dans `D:/Dev/model_gemma`** (ou modifiez le chemin dans le code)
3. **Ouvrez un terminal dans le dossier du projet**
4. **Exécutez le script d'installation automatique** :
   ```powershell
   python install_agrilens.py
   ```
5. **Lancez l'application** :
   ```powershell
   streamlit run src/streamlit_app_local_models.py
   ```

## 🇬🇧 Quick install
1. **Download or clone this repo**
2. **Place the Gemma 3n model in `D:/Dev/model_gemma`** (or modify the path in the code)
3. **Open a terminal in the project folder**
4. **Run the auto-install script**:
   ```powershell
   python install_agrilens.py
   ```
5. **Launch the app**:
   ```powershell
   streamlit run src/streamlit_app_local_models.py
   ```

---

## 🇫🇷 Nouvelles fonctionnalités (v2.0)

### 🎯 Modèle multimodal local
- **Gemma 3n multimodal** : Analyse simultanée image + texte
- **Paramètres optimisés** : Génération fluide sans caractères isolés
- **400 tokens** : Réponses détaillées et complètes

### 📱 Interface mobile
- **Design responsive** : Optimisé pour smartphones et tablettes
- **Sidebar collapsible** : Plus d'espace sur mobile
- **Boutons adaptés** : Taille et espacement optimisés
- **Feedback visuel** : Spinners et messages de statut

### 🔍 Diagnostic amélioré
- **Section automatique** : "Recommandations ou actions urgentes"
- **Analyse contextuelle** : Prise en compte de la question utilisateur
- **Conseils pratiques** : Actions prioritaires pour l'agriculteur

### ⚡ Performance
- **Chargement unique** : Modèle chargé une seule fois avec cache
- **CPU optimisé** : Fonctionne sans GPU
- **Feedback temps réel** : Indicateurs de progression

## 🇬🇧 New features (v2.0)

### 🎯 Local multimodal model
- **Gemma 3n multimodal** : Simultaneous image + text analysis
- **Optimized parameters** : Smooth generation without isolated characters
- **400 tokens** : Detailed and complete responses

### 📱 Mobile interface
- **Responsive design** : Optimized for smartphones and tablets
- **Collapsible sidebar** : More space on mobile
- **Adapted buttons** : Optimized size and spacing
- **Visual feedback** : Spinners and status messages

### 🔍 Enhanced diagnosis
- **Automatic section** : "Recommendations or urgent actions"
- **Contextual analysis** : User question consideration
- **Practical advice** : Priority actions for farmers

### ⚡ Performance
- **Single loading** : Model loaded once with cache
- **CPU optimized** : Works without GPU
- **Real-time feedback** : Progress indicators

---

## 🇫🇷 Script d'installation automatique
Le script `install_agrilens.py` :
- Crée l'environnement virtuel si besoin
- Installe toutes les dépendances (`requirements.txt`)
- Vérifie la présence du modèle
- Affiche les instructions de lancement

## 🇬🇧 Auto-install script
The `install_agrilens.py` script:
- Creates the virtual environment if needed
- Installs all dependencies (`requirements.txt`)
- Checks for the model presence
- Shows launch instructions

---

## 🇫🇷 Modes de fonctionnement

| Mode               | Modèle utilisé                        | Inférence réelle | Dépendance Internet | Remarques |
|-------------------|---------------------------------------|------------------|---------------------|-----------|
| **Local (offline)** | Gemma 3n multimodal (dossier local)   | ✅ Oui           | ❌ Non              | Rapide, 100% offline, recommandé |
| Hugging Face (token HF) | google/gemma-3n-E2B-it (API HF)         | ✅ Oui           | ✅ Oui              | Espace GPU recommandé, token requis |
| Hugging Face (public)   | Aucun (mode démo)                      | ❌ Non           | ✅ Oui              | Réponse factice, test UI uniquement |

### Instructions
- **Local (offline)** - **RECOMMANDÉ** :
  - Placez le modèle Gemma 3n dans `D:/Dev/model_gemma`
  - Lancez `streamlit run src/streamlit_app_local_models.py`
  - Aucun accès Internet requis
  - Interface mobile optimisée
- **Hugging Face (inférence réelle)** :
  - Ajoutez la variable d'environnement `HF_TOKEN`
  - Acceptez les conditions d'utilisation du modèle
  - Utilisez un Space GPU pour de meilleures performances
- **Hugging Face (mode démo)** :
  - Si aucun token n'est présent, mode démo uniquement

## 🇬🇧 Operating modes

| Mode               | Model used                            | Real inference   | Internet required   | Notes     |
|-------------------|---------------------------------------|------------------|---------------------|-----------|
| **Local (offline)** | Gemma 3n multimodal (local folder)    | ✅ Yes           | ❌ No               | Fast, 100% offline, recommended |
| Hugging Face (HF token) | google/gemma-3n-E2B-it (HF API)           | ✅ Yes           | ✅ Yes              | GPU Space recommended, token required |
| Hugging Face (public)   | None (demo mode)                         | ❌ No            | ✅ Yes              | Fictive answer, UI test only |

### Instructions
- **Local (offline)** - **RECOMMENDED** :
  - Put the Gemma 3n model in `D:/Dev/model_gemma`
  - Launch `streamlit run src/streamlit_app_local_models.py`
  - No Internet required
  - Mobile-optimized interface
- **Hugging Face (real inference)** :
  - Add the `HF_TOKEN` environment variable
  - Accept the model terms of use
  - Use a GPU Space for best performance
- **Hugging Face (demo mode)** :
  - If no token is present, demo mode only