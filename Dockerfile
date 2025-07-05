FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements d'abord pour le cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installation de PyTorch
RUN pip install --no-cache-dir --timeout=600 \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copie du code source
COPY . .

# Configuration des variables d'environnement
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Commande de démarrage
CMD ["python", "app.py"]