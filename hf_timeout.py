import os
import time
import requests

def wait_for_model_loading():
    """Attend que le modèle soit chargé"""
    max_retries = 30  # 30 tentatives maximum
    retry_delay = 60   # 60 secondes entre chaque tentative
    
    for i in range(max_retries):
        try:
            # Vérifie si l'application répond
            response = requests.get("http://localhost:8501/_stcore/health", timeout=10)
            if response.status_code == 200:
                print("L'application est prête !")
                return True
        except:
            pass
        
        print(f"En attente du chargement de l'application... (tentative {i+1}/{max_retries})")
        time.sleep(retry_delay)
    
    print("Délai d'attente dépassé. Vérifiez les logs pour plus d'informations.")
    return False

if __name__ == "__main__":
    wait_for_model_loading()