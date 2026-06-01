# Credit Risk Intelligence Platform
# ===================================
# Uses python:3.11-slim for a small, reproducible runtime image.
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# LightGBM requires OpenMP at runtime on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application source, models, data, and assets.
COPY . .

# FastAPI (8000) and Streamlit (8501) are exposed; compose selects the service command.
EXPOSE 8000 8501

# Default process runs the API; docker-compose overrides this for Streamlit.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
