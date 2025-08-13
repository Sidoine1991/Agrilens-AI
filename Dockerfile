FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl supervisor \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -i https://pypi.org/simple

# Code
COPY src/ /app/src/
COPY hf-optimized/ /app/hf-optimized/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENV MODEL_ID=google/gemma-3n-e2b-it DEVICE_MAP=auto MAX_NEW_TOKENS=256
EXPOSE 7860

# Healthcheck sur l’API (via 7860) – Streamlit est sur $PORT
HEALTHCHECK CMD sh -lc 'curl -fsS http://localhost:7860/health || exit 1'

CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]