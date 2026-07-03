FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# baixa o MobileSAM no build (não versionado no git), evitando latência no primeiro request
RUN mkdir -p pretrained_models && \
    python -c "from ultralytics import SAM; SAM('pretrained_models/mobile_sam.pt')"

COPY src/ src/
COPY static/ static/
COPY models/production/ models/production/

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
