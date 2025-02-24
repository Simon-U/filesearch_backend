# Stage 1: Build dependencies with Python 3.11
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04 AS builder

# Install system dependencies, PostgreSQL development libraries, and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git wget curl build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Manually install the latest pip version in the virtual environment
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Install setuptools and wheel separately to avoid broken dependencies
RUN python -m pip install --upgrade setuptools wheel

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
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install pydantic first to ensure correct version
RUN pip install --no-cache-dir "pydantic>=2.10.5,<3.0.0"

# Install key packages separately
RUN pip install --no-cache-dir \
    transformers==4.48.1 \
    Pillow==10.4.0 \
    pandas==2.2.3 \
    office365-rest-python-client==2.4.1 \
    boto3==1.35.63 \
    psycopg2==2.9.10

# Install the rest of requirements with retries
RUN pip install --no-cache-dir -r /tmp/requirements.txt || \
    (sleep 5 && pip install --no-cache-dir -r /tmp/requirements.txt) || \
    (sleep 10 && pip install --no-cache-dir -r /tmp/requirements.txt --no-deps)

# Remove the token from the requirements file for security
RUN sed -i 's|${GITHUB_TOKEN}|****|g' /tmp/requirements.txt

# Stage 2: Final lightweight image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

# Install Python 3.11 and required system libraries in the final stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment directory
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY app/services/fileloader /opt/ml/code/fileloader
COPY app/services/sagemaker/docker/inference.py /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code

# Optimize image size: remove apt caches and Python bytecode files
RUN rm -rf /var/lib/apt/lists/* \
    && find $VIRTUAL_ENV -type d -name "__pycache__" -exec rm -rf {} + \
    && find $VIRTUAL_ENV -type f -name "*.pyc" -delete \
    && find $VIRTUAL_ENV -type f -name "*.pyo" -delete

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