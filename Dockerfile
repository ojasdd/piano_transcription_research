# CPU-only container for Prior-Aware Piano Transcription
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    PORT=7860

# Install system dependencies (libsndfile for audio I/O, ffmpeg for decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 appuser
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app

# Copy dependency definition first for layer caching
COPY --chown=appuser:appuser requirements-deploy.txt .

# Install dependencies (CPU PyTorch and audio stack)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Expose Streamlit (7860) and FastAPI (8000) ports
EXPOSE 7860 8000

# Default entrypoint runs Streamlit UI on HF Spaces port 7860
# To run FastAPI instead: docker run -p 8000:8000 <image> uvicorn deployment.api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false"]
