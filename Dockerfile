FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && \
    apt-get install -y --no-install-recommends procps && \
    rm -rf /var/lib/apt/lists/*

# Création du répertoire src
RUN mkdir -p /app/src

# Copie des fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY src/ /app/src/
COPY start.sh /app/

# Définir les variables d'environnement
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Rendre le script exécutable
RUN chmod +x /app/start.sh

# Port exposé
EXPOSE 8501

# Commande de démarrage
CMD ["/app/start.sh"]