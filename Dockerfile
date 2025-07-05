FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \  # Pour la commande ps
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

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