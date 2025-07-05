FROM python:3.11-slim

WORKDIR /app

# 1. Mise à jour de pip seule
RUN pip install --upgrade pip

# 2. Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Installation de PyTorch séparément avec timeout augmenté
RUN pip install --no-cache-dir --timeout=600 \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# 5. Copie du code
COPY . .

# Configuration Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Port exposé
EXPOSE 8501

# Commande de démarrage
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]