FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Copier les fichiers
COPY requirements.txt ./
COPY src/ ./src/

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.org/simple

# Port FastAPI (valeur par défaut; sur Spaces, utilisez $PORT)
EXPOSE 7860

# Healthcheck FastAPI via le port effectif
HEALTHCHECK CMD sh -lc 'curl -fsS http://localhost:${PORT:-7860}/health || exit 1'

# Lancer l'API FastAPI sur le port fourni par HF ($PORT) ou 7860 par défaut
CMD ["sh","-lc","uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
