#!/bin/bash

# Chemin vers le répertoire de l'application
APP_DIR="/app"
LOG_FILE="/tmp/agrilens-startup.log"

# Fonction de log
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Se déplacer dans le répertoire de l'application
cd "$APP_DIR" || { log "Erreur: Impossible de se déplacer dans $APP_DIR"; exit 1; }

# Vérifier si le fichier streamlit_app.py existe
if [ ! -f "src/streamlit_app.py" ]; then
    log "Erreur: Le fichier src/streamlit_app.py est introuvable"
    log "Contenu du répertoire: $(ls -la)"
    log "Contenu de src/: $(ls -la src/ 2>&1)"
    exit 1
fi

# Vérifier si le token HF est défini
if [ -z "$HF_TOKEN" ]; then
    log "Avertissement: La variable d'environnement HF_TOKEN n'est pas définie"
fi

# Fonction pour démarrer Streamlit
start_streamlit() {
    log "Démarrage de Streamlit..."
    streamlit run src/streamlit_app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.fileWatcherType=none \
        --server.runOnSave=false \
        --server.headless=true \
        --browser.serverAddress=0.0.0.0 \
        --browser.gatherUsageStats=false \
        --logger.level=info
}

# Boucle de redémarrage
while true; do
    log "=== Démarrage de l'application ==="
    
    # Démarrer Streamlit en arrière-plan
    start_streamlit &
    STREAMLIT_PID=$!
    
    # Attendre que Streamlit démarre
    sleep 5
    
    # Vérifier si Streamlit est toujours en cours d'exécution
    if ! ps -p $STREAMLIT_PID > /dev/null; then
        log "Erreur: Streamlit n'a pas pu démarrer correctement"
        log "Dernières lignes du log:"
        tail -n 20 "$LOG_FILE"
    else
        log "Streamlit démarré avec succès (PID: $STREAMLIT_PID)"
    fi
    
    # Attendre que Streamlit se termine
    wait $STREAMLIT_PID
    STREAMLIT_EXIT_CODE=$?
    
    log "Streamlit s'est arrêté avec le code $STREAMLIT_EXIT_CODE"
    log "Redémarrage dans 2 secondes..."
    sleep 2
done