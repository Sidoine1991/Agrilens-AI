@echo off
echo 🌱 AgriLens AI - Version Locale
echo ================================
echo.

REM Vérifier si l'environnement virtuel existe
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Environnement virtuel non trouvé
    echo 💡 Créez-le avec : python -m venv venv
    pause
    exit /b 1
)

REM Activer l'environnement virtuel
echo 🔄 Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

REM Vérifier les dépendances
echo 📦 Vérification des dépendances...
python -c "import streamlit, transformers, torch, PIL, google.generativeai" 2>nul
if errorlevel 1 (
    echo ❌ Dépendances manquantes
    echo 💡 Installez-les avec : pip install -r requirements.txt
    pause
    exit /b 1
)

REM Configurer les variables d'environnement
echo 🔑 Configuration des API Keys...
set GOOGLE_API_KEY=AIzaSyC4a4z20p7EKq1Fk5_AX8eB_1yBo1HqYvA
set HF_TOKEN=hf_gUGRsgWffLNZVuzYLsmTdPwESIyrbryZW

echo ✅ Configuration terminée
echo 🚀 Lancement de l'application...
echo 📱 Interface web : http://localhost:8501
echo 🔄 Appuyez sur Ctrl+C pour arrêter
echo.

REM Lancer l'application
streamlit run src/streamlit_app_multilingual.py --server.port=8501 --server.address=localhost

pause 