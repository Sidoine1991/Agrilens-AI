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

# Port FastAPI
EXPOSE 7860

# Healthcheck FastAPI
HEALTHCHECK CMD curl --fail http://localhost:7860/health || exit 1

# Lancer l'API FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
