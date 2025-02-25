# Using official PyTorch image as base with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    libpq-dev \
    build-essential \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable for MMS
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Set CUDA environment variables
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
ENV FORCE_CUDA=1

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

# Install the SageMaker Inference Toolkit and Multi Model Server
RUN pip install --no-cache-dir multi-model-server sagemaker-inference

# Create directory for torch extensions cache
RUN mkdir -p /root/.cache/torch_extensions

# Create directory structure expected by SageMaker
RUN mkdir -p /opt/ml/model /opt/ml/code

# Copy application code
COPY app/services/fileloader /opt/ml/code/fileloader
COPY app/services/sagemaker/docker/inference.py /opt/ml/code/inference.py
COPY app/services/sagemaker/docker/serve.py /opt/ml/code/serve.py

# Set working directory
WORKDIR /opt/ml/code

# Optimize image size: remove Python bytecode files
RUN find / -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find / -type f -name "*.pyc" -delete \
    && find / -type f -name "*.pyo" -delete

# Ensure the application code is accessible
RUN chmod -R 755 /opt/ml/code

# Set Python path so modules are found
ENV PYTHONPATH=/opt/ml/code

# Set any default environment variables for your inference script if needed
ENV MODEL_TYPE=transformer
ENV MODEL_NAME=openai/clip-vit-base-patch32
ENV CONFIDENCE_THRESHOLD=0.4
ENV USE_HALF_PRECISION=true
ENV ENABLE_CAPTIONING=true
ENV CAPTION_MODEL=Salesforce/blip-image-captioning-base

# Set environment variables for SageMaker inference
ENV SAGEMAKER_PROGRAM=inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# Create an entrypoint script that starts the model server using serve.py when invoked with "serve"
RUN echo '#!/bin/bash\n\
    if [ "$1" = "serve" ]; then\n\
    exec python serve.py\n\
    else\n\
    exec "$@"\n\
    fi' > /usr/local/bin/serve && chmod +x /usr/local/bin/serve

# Set ENTRYPOINT to the serve script and default CMD to "serve"
ENTRYPOINT ["/usr/local/bin/serve"]
CMD ["serve"]