FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements d'abord pour utiliser le cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste de l'application
COPY . .

# Variables d'environnement
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV TRANSFORMERS_OFFLINE=0
ENV HF_HUB_OFFLINE=0

# Port exposé
EXPOSE 8501

# Commande de démarrage avec timeout augmenté
CMD ["streamlit", "run", "src/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none", \
     "--server.runOnSave=false", \
     "--server.headless=true"]