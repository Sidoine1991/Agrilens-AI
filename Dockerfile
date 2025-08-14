FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements d'abord pour le cache Docker
COPY requirements.txt /app/requirements.txt

# Installer PyTorch et torchvision AVANT le reste
RUN pip install --no-cache-dir --default-timeout=600 \
    torch>=2.1.0 torchvision>=0.16.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Installer timm pour les modèles de vision
RUN pip install --no-cache-dir timm>=0.9.0

# Installer les dépendances pour les modèles de vision
RUN pip install --no-cache-dir pillow>=10.0.0

# Installer FastAPI et uvicorn
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# Installer le reste des dépendances
RUN pip install --no-cache-dir --default-timeout=600 -r requirements.txt -i https://pypi.org/simple

# Copie du code source
COPY src/ /app/src/

# Créer le dossier cache Hugging Face et donner les droits d'écriture
RUN mkdir -p /app/cache/huggingface && chmod -R 777 /app/cache/huggingface
ENV HF_HOME=/app/cache/huggingface

# Variables d'environnement
ENV PORT=7860
ENV HOST=0.0.0.0
ENV MODEL_ID=google/gemma-3n-e2b-it
ENV DEVICE_MAP=auto
ENV MAX_NEW_TOKENS=256

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Commande de démarrage FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]