# =================================================================
# Dockerfile — Driver Activity Detection Prediction Service
# =================================================================
# Build:   docker build -t driver-activity-service .
# Run:     docker run -p 5000:5000 driver-activity-service
# =================================================================

FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, templates, and model
COPY app.py .
COPY templates/ templates/
COPY model/ model/

# Expose the Flask port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
