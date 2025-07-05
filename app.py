import subprocess
import os
import sys

print("=== Démarrage du serveur Streamlit ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print("Contenu du répertoire courant:")
os.system("ls -la")
print("\nContenu du dossier src:")
os.system("ls -la src/")

try:
    os.chdir("src")
    print("\n=== Lancement de Streamlit ===")
    subprocess.run([
        "streamlit", 
        "run", 
        "streamlit_app.py", 
        "--server.port=8501", 
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ], check=True)
except Exception as e:
    print(f"Erreur: {str(e)}")
    print("\n=== Stack Trace ===")
    import traceback
    traceback.print_exc()
    # Garder le conteneur en vie pour voir les logs
    input("Appuyez sur Entrée pour quitter...")