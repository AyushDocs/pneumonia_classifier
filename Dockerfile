# Stage 1: Build Dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Optimize layer caching: install requirements first (ignoring editable install)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    grep -v "^-e" requirements.txt > requirements_cache.txt && \
    pip install --no-cache-dir --user -r requirements_cache.txt

# Stage 2: Final Image
FROM python:3.10-slim

# Create a non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder to the appuser's home
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUSERBASE=/home/appuser/.local

# Copy models first (heavy but infrequent changes)
COPY model_dir/ /app/model_dir/
COPY models/ /app/models/

# Copy application code
COPY app.py streamlit_app.py setup.py requirements.txt ./
COPY pneumonia_classifier/ /app/pneumonia_classifier/
COPY frontend/ /app/frontend/
COPY prometheus/ /app/prometheus/
COPY samples/ /app/samples/
COPY static/ /app/static/

# Ensure internal directories exist and set permissions
RUN mkdir -p logs artifacts data/heatmaps data/samples && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
