
#!/bin/bash
cd /app
streamlit run src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.fileWatcherType none --server.runOnSave false --server.headless true
# Démarrer Streamlit en arrière-plan

# Vérification simplifiée
while true; do
    # Vérifier si le processus Streamlit est toujours en cours
    if ! ps -p $! > /dev/null; then
        echo "Le serveur Streamlit a été arrêté. Redémarrage..."
        streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.fileWatcherType none --server.runOnSave false --server.headless true &
    fi
    sleep 30
done