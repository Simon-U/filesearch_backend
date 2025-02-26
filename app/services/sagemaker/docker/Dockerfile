# Using official PyTorch image as base with CUDA support
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
ENV FORCE_CUDA=1

# Verify Python version
RUN python3 --version | grep -E "3\.10\.|3\.11\." || (echo "Unsupported Python version" && exit 1)

# Install system dependencies including libssl-dev for OpenSSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    libpq-dev \
    build-essential \
    libssl-dev \
    openjdk-11-jre-headless \
    g++ \
    make \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable for MMS
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Accept GitHub token as a build argument
ARG GITHUB_TOKEN

# Configure Git to use HTTPS instead of SSH with token support
RUN git config --global url."https://".insteadOf git:// && \
    git config --global url."https://".insteadOf ssh:// && \
    git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

# Build torchvision from source without CUDA to avoid CUDA_HOME issues
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    # Create a small test script to verify installation
    python -c "import torch; import torchvision; print(f'PyTorch: {torch.__version__}, torchvision: {torchvision.__version__}, CUDA available: {torch.cuda.is_available()}')"
# Copy the requirements file from the repository root
COPY requirements.txt /tmp/requirements.txt

# Install requirements in one pass with error handling
RUN pip install --no-cache-dir -r /tmp/requirements.txt || \
    (sleep 5 && pip install --no-cache-dir -r /tmp/requirements.txt) || \
    (sleep 10 && pip install --no-cache-dir -r /tmp/requirements.txt) || \
    (echo "Failed to install requirements" && exit 1)



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

# Add a diagnostic script to verify CUDA setup
RUN echo '#!/bin/bash\n\
    echo "NVIDIA Driver:"\n\
    nvidia-smi\n\
    echo ""\n\
    echo "PyTorch CUDA Information:"\n\
    python -c "import torch; print(f\"PyTorch version: {torch.__version__}\"); \
    print(f\"CUDA available: {torch.cuda.is_available()}\"); \
    print(f\"CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}\"); \
    print(f\"GPU Count: {torch.cuda.device_count()}\"); \
    print(f\"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}\")"\n\
    ' > /usr/local/bin/check_cuda && chmod +x /usr/local/bin/check_cuda

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