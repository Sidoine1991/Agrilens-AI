#!/bin/bash
# AgriLens AI - Script d'installation et de lancement automatisé
# 🇫🇷 Ce script prépare l'environnement et lance l'application (Linux/Mac)
# 🇬🇧 This script sets up the environment and launches the app (Linux/Mac)

set -e

# Vérification du modèle local
MODEL_DIR="models/gemma-3n"
if [ ! -d "$MODEL_DIR" ]; then
  echo "[FR] Le dossier du modèle ($MODEL_DIR) est manquant. Placez les fichiers Gemma 3n dans ce dossier."
  echo "[EN] Model folder ($MODEL_DIR) is missing. Please put Gemma 3n files in this folder."
  exit 1
fi

# Création de l'environnement virtuel
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate

# Installation des dépendances
pip install --upgrade pip
pip install -r requirements.txt

# Lancement de l'application
streamlit run src/streamlit_app.py