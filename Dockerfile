FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configuration de pip
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=5

# Installation des dépendances une par une avec des retries
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=300 --retries 5 \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copie des requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 --retries 5 -r requirements.txt

# Copie du code
COPY . .

# Variables d'environnement
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Création du répertoire src
RUN mkdir -p /app/src

# Port exposé
EXPOSE 8501

# Commande de démarrage
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]