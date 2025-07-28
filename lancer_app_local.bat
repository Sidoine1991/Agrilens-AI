@echo off
echo 🌱 AgriLens AI - Version Locale avec Modèle Local
echo ==================================================
echo.

REM Vérifier si l'environnement virtuel existe
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Environnement virtuel non trouvé
    echo 💡 Créez-le avec : python -m venv venv
    pause
    exit /b 1
)

REM Vérifier si le modèle local existe
if not exist "D:\Dev\model_gemma" (
    echo ❌ Modèle local non trouvé dans D:\Dev\model_gemma
    echo 💡 Assurez-vous que le modèle Gemma 3n E4B IT est téléchargé dans ce dossier
    pause
    exit /b 1
)

REM Activer l'environnement virtuel
echo 🔄 Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

REM Configurer les variables d'environnement
echo 🔑 Configuration des API Keys...
set GOOGLE_API_KEY=AIzaSyC4a4z20p7EKq1Fk5_AX8eB_1yBo1HqYvA
set HF_TOKEN=hf_gUGRsgWffLNZVuzYLsmTdPwESIyrbryZW

echo ✅ Configuration terminée
echo 🚀 Lancement de l'application avec modèle local...
echo 📁 Modèle : D:\Dev\model_gemma
echo 📱 Interface web : http://localhost:8501
echo 🔄 Appuyez sur Ctrl+C pour arrêter
echo.

REM Lancer l'application avec le modèle local
streamlit run src/streamlit_app_multilingual.py --server.port=8501 --server.address=localhost

pause 