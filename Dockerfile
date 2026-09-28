FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install CPU-only PyTorch first so the requirements below don't pull the CUDA wheels
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Copy and install Python dependencies first (for Docker layer caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as an unprivileged user
RUN useradd --create-home --uid 1000 appuser

# Copy project files (includes model_checkpoint.pth, see .dockerignore)
COPY --chown=appuser:appuser . .

# Create results directory
RUN mkdir -p /app/results && chown appuser:appuser /app/results

USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check (python:3.11-slim has no curl, so use the standard library)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5).status == 200 else 1)"

# Default: run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
