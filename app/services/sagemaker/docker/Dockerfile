# Stage 1: Build dependencies
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker AS builder

# Install only necessary dependencies
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    torch==2.0.1 \
    Pillow==10.0.0 \
    pandas==2.1.0 \
    office365-rest-python-client==2.4.1 \
    boto3==1.28.0

# Stage 2: Final lightweight image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy application code
COPY fileloader /opt/ml/code/fileloader
COPY sagemaker/docker/inference.py /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code

# Optimize image size
RUN rm -rf /var/lib/apt/lists/* \
    && find /usr/local/lib/python3.10/site-packages -type d -name "__pycache__" -exec rm -rf {} + \
    && find /usr/local/lib/python3.10/site-packages -type f -name "*.pyc" -delete \
    && find /usr/local/lib/python3.10/site-packages -type f -name "*.pyo" -delete

# Make sure files are accessible
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