# Stage 1: Build a base image with Python 3.11 and CUDA support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ensure Python 3.11 is the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip, setuptools, and wheel to ensure smooth metadata generation
RUN python -m pip install --upgrade pip setuptools wheel

# Accept GitHub token as a build argument for private repositories
ARG GITHUB_TOKEN

# Copy the requirements file from the repository root
COPY requirements.txt /tmp/requirements.txt

# Replace placeholder in requirements.txt with the actual GitHub token
RUN if [ -n "$GITHUB_TOKEN" ]; then \
    sed -i "s|__GITHUB_TOKEN__|$GITHUB_TOKEN|g" /tmp/requirements.txt; \
    fi

# Install PyTorch with CUDA support & other dependencies
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir \
    transformers==4.36.0 \
    Pillow==10.0.0 \
    pandas==2.1.0 \
    office365-rest-python-client==2.4.1 \
    boto3==1.28.0

# Stage 2: Final lightweight image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY app/services/fileloader /opt/ml/code/fileloader
COPY app/services/sagemaker/docker/inference.py /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code

# Optimize image size: remove apt caches and Python bytecode files
RUN rm -rf /var/lib/apt/lists/* \
    && find /usr/local/lib/python3.11/site-packages -type d -name "__pycache__" -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages -type f -name "*.pyc" -delete \
    && find /usr/local/lib/python3.11/site-packages -type f -name "*.pyo" -delete

# Ensure the application code is accessible
RUN chmod -R 755 /opt/ml/code

# Set Python path so modules are found
ENV PYTHONPATH=/opt/ml/code

# Set default environment variables for analyzer
ENV MODEL_TYPE=transformer
ENV MODEL_NAME=openai/clip-vit-base-patch32
ENV CONFIDENCE_THRESHOLD=0.4
ENV USE_HALF_PRECISION=true
ENV ENABLE_CAPTIONING=true
ENV CAPTION_MODEL=Salesforce/blip-image-captioning-base

# Default command to run your inference script
CMD ["python", "inference.py"]
