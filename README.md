---
title: AgriLens AI
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
sdk_version: "1.0.0"
app_file: src/streamlit_app.py
pinned: false
---

# AgriLens AI 🌱

Application de diagnostic des maladies des plantes utilisant le modèle Gemma 3n de Google.

## Fonctionnalités
- Analyse d'images de plantes
- Détection des maladies
- Recommandations de traitement personnalisées

## Comment utiliser
1. Téléchargez une photo d'une plante
2. Cliquez sur "Analyser l'image"
3. Consultez les résultats détaillés

## Configuration requise
- Hardware: GPU recommandé
- Espace disque: 20GB+ (pour le modèle Gemma 3n)

## Avertissement
Les résultats fournis sont à titre indicatif uniquement. Pour un diagnostic médical ou agricole professionnel, veuillez consulter un expert qualifié.

## Développement
- Framework: Streamlit
- Modèle: google/gemma-3n-e4b-it
- Dernière mise à jour: Juillet 2025

---

## 🇫🇷 Installation rapide
1. **Téléchargez ou clonez ce dépôt**
2. **Placez le dossier du modèle Gemma 3n dans `models/`** (exemple : `models/gemma-3n-transformers-gemma-3n-e2b-it-v1`)
3. **Ouvrez un terminal dans le dossier du projet**
4. **Exécutez le script d’installation automatique** :
   ```powershell
   python install_agrilens.py
   ```
5. **Lancez l’application** :
   ```powershell
   streamlit run src/streamlit_app.py
   ```

## 🇬🇧 Quick install
1. **Download or clone this repo**
2. **Place the Gemma 3n model folder in `models/`** (e.g. `models/gemma-3n-transformers-gemma-3n-e2b-it-v1`)
3. **Open a terminal in the project folder**
4. **Run the auto-install script**:
   ```powershell
   python install_agrilens.py
   ```
5. **Launch the app**:
   ```powershell
   streamlit run src/streamlit_app.py
   ```

---

## 🇫🇷 Script d’installation automatique
Le script `install_agrilens.py` :
- Crée l’environnement virtuel si besoin
- Installe toutes les dépendances (`requirements.txt`)
- Vérifie la présence du modèle dans `models/`
- Affiche les instructions de lancement

## 🇬🇧 Auto-install script
The `install_agrilens.py` script:
- Creates the virtual environment if needed
- Installs all dependencies (`requirements.txt`)
- Checks for the model in `models/`
- Shows launch instructions