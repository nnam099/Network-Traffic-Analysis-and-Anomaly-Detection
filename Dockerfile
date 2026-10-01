FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    IDS_MODEL_PATH=/app/checkpoints/ids_v14_model.pth \
    IDS_PIPELINE_PATH=/app/checkpoints/ids_v14_pipeline.pkl

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --progress-bar off -r requirements.txt

COPY . .
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/results \
    && chown -R appuser:appuser /app/results

EXPOSE 8080

USER appuser

CMD ["uvicorn", "ids.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8080"]
