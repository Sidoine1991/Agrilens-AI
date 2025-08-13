FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gnupg \
    lsb-release \
 && rm -rf /var/lib/apt/lists/*

# Copier les fichiers
COPY requirements.txt ./
COPY src/ ./src/

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port Streamlit
EXPOSE 8501

# Healthcheck pour vérifier si Streamlit tourne
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Lancer l'application Streamlit
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
