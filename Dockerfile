FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -i https://pypi.org/simple

# Code
COPY src/ /app/src/

# Variables par défaut (peuvent être surchargées par Secrets/Variables du Space)
ENV MODEL_ID=google/gemma-3n-e2b-it \
    DEVICE_MAP=auto \
    MAX_NEW_TOKENS=256

# Port (Spaces injecte $PORT)
EXPOSE 7860

# Healthcheck sur le port effectif
HEALTHCHECK CMD sh -lc 'curl -fsS http://localhost:${PORT:-7860}/health || exit 1'

# Lancer uvicorn sur le port fourni par Spaces
CMD ["sh","-lc","uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-7860}"]