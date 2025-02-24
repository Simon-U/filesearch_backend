# Stage 1: Build dependencies
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime-py311 AS builder

# Install git so that pip can clone private repositories
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Accept GitHub token as a build argument
ARG GITHUB_TOKEN

# Copy the requirements file from the repository root
COPY requirements.txt /tmp/requirements.txt

# Replace placeholder in requirements.txt with the actual GitHub token
RUN if [ -n "$GITHUB_TOKEN" ]; then \
    sed -i "s|__GITHUB_TOKEN__|$GITHUB_TOKEN|g" /tmp/requirements.txt; \
    fi

# Upgrade pip, setuptools, and wheel to ensure smooth metadata generation
RUN pip install --upgrade pip setuptools wheel

# Install dependencies (this will use git+https URLs for private repos)
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir \
    transformers==4.36.0 \
    torch==2.0.1 \
    Pillow==10.0.0 \
    pandas==2.1.0 \
    office365-rest-python-client==2.4.1 \
    boto3==1.28.0

# Stage 2: Final lightweight image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime-py311

# Copy installed Python packages from builder
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages

# Copy application code
COPY app/services/fileloader /opt/ml/code/fileloader
COPY app/services/sagemaker/docker/inference.py /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code

# Optimize image size: remove apt caches and Python bytecode files
RUN rm -rf /var/lib/apt/lists/* \
    && find /opt/conda/lib/python3.10/site-packages -type d -name "__pycache__" -exec rm -rf {} + \
    && find /opt/conda/lib/python3.10/site-packages -type f -name "*.pyc" -delete \
    && find /opt/conda/lib/python3.10/site-packages -type f -name "*.pyo" -delete

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