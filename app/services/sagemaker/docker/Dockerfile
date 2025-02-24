# Stage 1: Build dependencies
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS builder

# Install git so that pip can clone private repositories
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Accept SSH private key as a build argument
ARG SSH_PRIVATE_KEY

# If the SSH key is provided, set up SSH
RUN if [ -n "$SSH_PRIVATE_KEY" ]; then \
    mkdir -p /root/.ssh && \
    echo "$SSH_PRIVATE_KEY" > /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts; \
    fi

# Copy the requirements file from the repository root
COPY requirements.txt /tmp/requirements.txt

# Install dependencies (this will use git+ssh URLs for private repos)
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir \
    transformers==4.36.0 \
    torch==2.0.1 \
    Pillow==10.0.0 \
    pandas==2.1.0 \
    office365-rest-python-client==2.4.1 \
    boto3==1.28.0

# Remove the SSH key so it doesn’t end up in later stages
RUN rm -f /root/.ssh/id_rsa

# Stage 2: Final lightweight image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Copy installed Python packages from builder
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages

# Copy application code
COPY app/services/fileloader /opt/ml/code/fileloader
COPY app/services/sagemaker/docker/inference.py /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code

# Optimize image size
RUN rm -rf /var/lib/apt/lists/* \
    && find /opt/conda/lib/python3.10/site-packages -type d -name "__pycache__" -exec rm -rf {} + \
    && find /opt/conda/lib/python3.10/site-packages -type f -name "*.pyc" -delete \
    && find /opt/conda/lib/python3.10/site-packages -type f -name "*.pyo" -delete

# Ensure application code is accessible
RUN chmod -R 755 /opt/ml/code

# Set Python path
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
