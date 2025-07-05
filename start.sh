#!/bin/bash

# Démarrer Streamlit en arrière-plan
streamlit run src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.fileWatcherType none --server.runOnSave false --server.headless true &

# Vérifier périodiquement que l'application est en cours d'exécution
while true; do
    # Vérifier si le serveur est en cours d'exécution
    if ! pgrep -f "streamlit" > /dev/null; then
        echo "Le serveur Streamlit a été arrêté. Redémarrage..."
        streamlit run src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.fileWatcherType none --server.runOnSave false --server.headless true &
    fi
    sleep 30
done