# Stage 1: Build dependencies with Python 3.11
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, PostgreSQL development libraries, and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    python3-setuptools \
    git \
    wget \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create directory for virtual environment
RUN mkdir -p /opt/venv

# Create virtual environment
RUN python3.11 -m venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in the virtual environment
RUN /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

# Accept GitHub token as a build argument
ARG GITHUB_TOKEN

# Configure Git to use HTTPS instead of SSH
RUN git config --global url."https://".insteadOf git:// && \
    git config --global url."https://".insteadOf ssh:// && \
    git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

# Copy the requirements file from the repository root
COPY requirements.txt /tmp/requirements.txt

# Replace placeholder in requirements.txt with the actual GitHub token
RUN if [ -n "$GITHUB_TOKEN" ]; then \
    sed -i "s|__GITHUB_TOKEN__|${GITHUB_TOKEN}|g" /tmp/requirements.txt; \
    fi

# Install base dependencies first
RUN /opt/venv/bin/pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install pydantic first to ensure correct version
RUN /opt/venv/bin/pip install --no-cache-dir "pydantic>=2.10.5,<3.0.0"

# Install key packages separately
RUN /opt/venv/bin/pip install --no-cache-dir \
    transformers==4.48.1 \
    Pillow==10.4.0 \
    pandas==2.2.3 \
    office365-rest-python-client==2.4.1 \
    boto3==1.35.63 \
    psycopg2==2.9.10

# Install the rest of requirements with retries
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt || \
    (sleep 5 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt) || \
    (sleep 10 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt --no-deps)

# Remove the token from the requirements file for security
RUN sed -i 's|${GITHUB_TOKEN}|****|g' /tmp/requirements.txt

# Stage 2: Final lightweight image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and required system libraries in the final stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment to use virtual environment and ensure /usr/local/bin is in the PATH
ENV PATH="/usr/local/bin:/opt/venv/bin:$PATH"

# Copy application code
COPY app/services/fileloader /opt/ml/code/fileloader
COPY app/services/sagemaker/docker/inference.py /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code

# Optimize image size: remove apt caches and Python bytecode files
RUN rm -rf /var/lib/apt/lists/* \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + \
    && find /opt/venv -type f -name "*.pyc" -delete \
    && find /opt/venv -type f -name "*.pyo" -delete

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

# Install SageMaker serving toolkit
RUN /opt/venv/bin/pip install --no-cache-dir sagemaker-inference flask

# Create directory structure expected by SageMaker
RUN mkdir -p /opt/ml/model

# Create a wrapper shell script to serve as the entry point for SageMaker
RUN echo '#!/bin/bash\n\
    if [ "$1" = "serve" ]; then\n\
    python -m sagemaker_inference.model_server.start_model_server\n\
    else\n\
    eval "$@"\n\
    fi' > /usr/local/bin/serve && chmod +x /usr/local/bin/serve

# Set environment variables for SageMaker
ENV SAGEMAKER_PROGRAM=inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_HANDLER=inference.model_fn

# Override the NVIDIA entrypoint to use our serve script
ENV NVIDIA_DISABLE_REQUIRE=true

# Clear inherited entrypoint and use CMD instead
ENTRYPOINT []
CMD ["serve", "serve"]