# CBIR Backend - Docker Image
# Python 3.11 with all dependencies for the search engine

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy frontend (will be served by backend)
COPY frontend/ ./frontend/

# Set environment variables
ENV MONGO_URI=mongodb://mongo:27017
ENV DATA_DIR=/app/data
ENV TORCH_HOME=/app/data/models/torch
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=5)" || exit 1

# Prepare missing data assets, then start the API without the development reloader.
CMD ["sh", "-c", "python backend/scripts/prepare_data.py && cd backend && exec uvicorn main:app --host 0.0.0.0 --port 8000"]
