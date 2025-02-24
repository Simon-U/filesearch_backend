# Use AWS PyTorch 2.2 container with Python 3.11 and CUDA 12.1
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py311-cu121-ubuntu20.04-sagemaker

# Accept GitHub token as a build argument
ARG GITHUB_TOKEN

# Configure Git to use HTTPS instead of SSH when the token is provided
RUN if [ -n "$GITHUB_TOKEN" ]; then \
    git config --global url."https://".insteadOf git:// && \
    git config --global url."https://".insteadOf ssh:// && \
    git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi

# Copy the requirements file (assuming it's in the build context)
COPY requirements.txt /tmp/requirements.txt

# Replace placeholder in requirements.txt with the actual GitHub token if needed
RUN if [ -n "$GITHUB_TOKEN" ]; then \
    sed -i "s|__GITHUB_TOKEN__|${GITHUB_TOKEN}|g" /tmp/requirements.txt; \
    fi

# Install key dependencies
# Note: PyTorch is already installed in the base image
RUN pip install --no-cache-dir \
    "pydantic>=2.10.5,<3.0.0" \
    transformers==4.48.1 \
    Pillow==10.4.0 \
    pandas==2.2.3 \
    office365-rest-python-client==2.4.1 \
    boto3==1.35.63 \
    psycopg2==2.9.10

# Install remaining packages from requirements.txt with retries
# Excluding packages that may conflict with the base image
RUN pip install --no-cache-dir -r /tmp/requirements.txt --ignore-installed || \
    (sleep 5 && pip install --no-cache-dir -r /tmp/requirements.txt --ignore-installed) || \
    (sleep 10 && pip install --no-cache-dir -r /tmp/requirements.txt --ignore-installed --no-deps)

# Remove the token from the requirements file for security
RUN if [ -n "$GITHUB_TOKEN" ]; then \
    sed -i 's|${GITHUB_TOKEN}|****|g' /tmp/requirements.txt; \
    fi

# Copy application code
COPY app/services/fileloader /opt/ml/code/fileloader
COPY app/services/sagemaker/docker/inference.py /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code

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

# Install SageMaker inference package if not already installed
RUN pip install --no-cache-dir sagemaker-inference flask

# Set environment variables for SageMaker
ENV SAGEMAKER_PROGRAM=inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_HANDLER=inference.model_fn

# The AWS DLC should already have appropriate entrypoint setup for SageMaker
# but we can add a custom serve script if needed
RUN if [ ! -f /usr/local/bin/serve ]; then \
    echo '#!/bin/bash\n\
    if [ "$1" = "serve" ]; then\n\
    exec python -m sagemaker_inference.model_server.start_model_server\n\
    else\n\
    exec "$@"\n\
    fi' > /usr/local/bin/serve && chmod +x /usr/local/bin/serve; \
    fi

# Entrypoint and CMD may be inherited from base image
# If you need to customize:
# ENTRYPOINT ["/usr/local/bin/serve"]
# CMD ["serve"]