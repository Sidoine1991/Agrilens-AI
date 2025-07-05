FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements d'abord pour le cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installer torch/torchvision et opencv-python-headless AVANT le reste
RUN pip install --no-cache-dir --default-timeout=600 \
    torch==2.0.1+cpu torchvision==0.15.2+cpu opencv-python-headless==4.11.0.86 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Installer le reste des dépendances
RUN pip install --no-cache-dir --default-timeout=600 -r requirements.txt -i https://pypi.org/simple

# Copie du code source
COPY . .

# Créer le dossier cache Hugging Face et donner les droits d'écriture
RUN mkdir -p /app/cache/huggingface && chmod -R 777 /app/cache/huggingface
ENV HF_HOME=/app/cache/huggingface

# Configuration Streamlit
# Pas besoin de variables d'environnement spécifiques pour Hugging Face Spaces
WORKDIR /app

# Commande de démarrage
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]