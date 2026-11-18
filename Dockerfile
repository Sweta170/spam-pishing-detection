# Use official lightweight Python base image
FROM python:3.10-slim

# Set container working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable instant stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies without caching wheels to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and trained models
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .
COPY download_data.py .

# Expose FastAPI default port
EXPOSE 8000

# Healthcheck to verify service availability
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start the application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
