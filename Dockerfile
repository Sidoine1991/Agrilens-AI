FROM python:3.11-slim

# Répertoire de travail
WORKDIR /app

# Installer dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.org/simple

# Copier le code de l'application
COPY . .

# Variables d'environnement
ENV MODEL_ID=google/gemma-3n-e2b-it \
    DEVICE_MAP=auto \
    MAX_NEW_TOKENS=256 \
    PORT=7860

# Exposer le port (Hugging Face redirige vers $PORT)
EXPOSE ${PORT}

# Lancement Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=${PORT}", "--server.address=0.0.0.0"]
