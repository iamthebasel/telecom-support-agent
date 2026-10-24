FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only torch keeps the image small; the rest comes from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# run from source (PYTHONPATH) so data paths resolve relative to /app
COPY src ./src
COPY data/knowledge_base ./data/knowledge_base
COPY eval ./eval

# Build the vector store and customer DB at image build time (downloads the embedding model once)
RUN python -m telecom_agent seed-db && python -m telecom_agent ingest

EXPOSE 8000
CMD ["python", "-m", "telecom_agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
