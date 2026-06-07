FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY web/ ./web/
# Set API_URL to use nginx reverse proxy path for Docker
RUN python -c "p='web/app.js'; t=open(p).read(); open(p,'w').write(t.replace(\"'http://127.0.0.1:8001'\", \"'/api'\"))"
COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create knowledge_bases directory (will be overridden by volume)
RUN mkdir -p /app/knowledge_bases

EXPOSE 80

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
