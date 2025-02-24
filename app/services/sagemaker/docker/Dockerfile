# Stage 1: Build dependencies with Ubuntu 20.04 and Python 3.10
FROM ubuntu:20.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    gnupg \
    ca-certificates \
    git \
    curl \
    build-essential \
    libpq-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    python3-setuptools

# Install CUDA toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-toolkit-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV PATH="/usr/local/cuda-12.1/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}"

# Create directory for virtual environment
RUN mkdir -p /opt/venv

# Create virtual environment using Python 3.10
RUN python3.10 -m venv /opt/venv

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

# Install base dependencies first (with appropriate versions)
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

# Stage 2: Final lightweight image using Ubuntu 20.04 and Python 3.10
FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and required system libraries in the final stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    gnupg \
    ca-certificates \
    libpq5 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv

# Install CUDA runtime only (not full toolkit)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-runtime-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV PATH="/usr/local/cuda-12.1/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}"

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment to use virtual environment
ENV PATH="/opt/venv/bin:${PATH}"

# Copy application code; update to use the training script
COPY app/services/fileloader /opt/ml/code/fileloader
COPY app/services/sagemaker/docker/inference.py /opt/ml/code/train.py

# Set working directory
WORKDIR /opt/ml/code

# Optimize image size: remove apt caches and Python bytecode files
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type f -name "*.pyc" -delete \
    && find /opt/venv -type f -name "*.pyo" -delete

# Ensure the application code is accessible
RUN chmod -R 755 /opt/ml/code

# Set Python path so modules are found
ENV PYTHONPATH=/opt/ml/code

# Set any default environment variables for your training script if needed
ENV MODEL_TYPE=transformer
ENV MODEL_NAME=openai/clip-vit-base-patch32
ENV CONFIDENCE_THRESHOLD=0.4
ENV USE_HALF_PRECISION=true
ENV ENABLE_CAPTIONING=true
ENV CAPTION_MODEL=Salesforce/blip-image-captioning-base

# Install the SageMaker Training Toolkit
RUN /opt/venv/bin/pip install --no-cache-dir sagemaker-training

# Create directory structure expected by SageMaker
RUN mkdir -p /opt/ml/model

# Set environment variables for SageMaker training
ENV SAGEMAKER_PROGRAM=train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# Use the training script as the container's entry point
CMD ["python", "/opt/ml/code/train.py"]