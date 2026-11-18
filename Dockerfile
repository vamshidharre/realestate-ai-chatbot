# Runs on any Docker host. Render passes the port in $PORT; locally it defaults to 7860.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    ANONYMIZED_TELEMETRY=False \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && chown user:user /app

# Everything from here runs as the non-root user, so no extra chown layer is needed.
USER user
COPY --chown=user:user . .

# Build the vector store into the image from the committed scrape snapshot (data/pages.json).
# This also downloads the ONNX embedding model into $HOME/.cache/chroma, so the container starts fast.
RUN python scraper_indexer.py

EXPOSE 7860
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
