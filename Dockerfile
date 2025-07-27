FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements d'abord pour le cache Docker
COPY requirements.txt .

# Installer PyTorch et torchvision AVANT le reste
RUN pip install --no-cache-dir --default-timeout=600 \
    torch>=2.1.0 torchvision>=0.16.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Installer le reste des dépendances
RUN pip install --no-cache-dir --default-timeout=600 -r requirements.txt -i https://pypi.org/simple

# Copie du code source
COPY . .

# Créer le dossier cache Hugging Face et donner les droits d'écriture
RUN mkdir -p /app/cache/huggingface && chmod -R 777 /app/cache/huggingface
ENV HF_HOME=/app/cache/huggingface

# Configuration Streamlit
WORKDIR /app

# Commande de démarrage - VERSION SIMPLIFIÉE
CMD ["streamlit", "run", "src/streamlit_app_simple.py", "--server.port=7860", "--server.address=0.0.0.0"]