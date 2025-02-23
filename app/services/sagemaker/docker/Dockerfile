# Stage 1: Build dependencies
# Using a public PyTorch base image. Adjust the tag if needed.
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS builder

# Install only necessary Python dependencies
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    torch==2.0.1 \
    Pillow==10.0.0 \
    pandas==2.1.0 \
    office365-rest-python-client==2.4.1 \
    boto3==1.28.0

# Stage 2: Final lightweight image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Copy only necessary packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy application code into the container
COPY app/services/fileloader /opt/ml/code/fileloader
COPY app/services/sagemaker/docker/inference.py /opt/ml/code/

# Set the working directory
WORKDIR /opt/ml/code

# Optimize image size by removing unnecessary caches and bytecode files
RUN rm -rf /var/lib/apt/lists/* \
    && find /usr/local/lib/python3.10/site-packages -type d -name "__pycache__" -exec rm -rf {} + \
    && find /usr/local/lib/python3.10/site-packages -type f -name "*.pyc" -delete \
    && find /usr/local/lib/python3.10/site-packages -type f -name "*.pyo" -delete

# Ensure files are accessible
RUN chmod -R 755 /opt/ml/code

# Set Python path so your modules are found
ENV PYTHONPATH=/opt/ml/code

# Set default environment variables for the analyzer
ENV MODEL_TYPE=transformer
ENV MODEL_NAME=openai/clip-vit-base-patch32
ENV CONFIDENCE_THRESHOLD=0.4
ENV USE_HALF_PRECISION=true
ENV ENABLE_CAPTIONING=true
ENV CAPTION_MODEL=Salesforce/blip-image-captioning-base

# Default command to run your inference script
CMD ["python", "inference.py"]