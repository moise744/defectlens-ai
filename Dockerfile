# DefectLens AI — Dockerfile
FROM python:3.11-slim

LABEL maintainer="Habimana Moise <habimanamoise77@gmail.com>"
LABEL description="DefectLens AI: Industrial Visual Inspection & Anomaly Detection"

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only PyTorch to keep image size manageable)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt fastapi uvicorn[standard] python-multipart

# Copy source code
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY scripts/ scripts/
COPY setup.py .

RUN pip install --no-cache-dir -e .

# Streamlit port
EXPOSE 8501
# FastAPI port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Default: run Streamlit dashboard
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
