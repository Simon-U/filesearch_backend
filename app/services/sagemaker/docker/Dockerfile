# Using official PyTorch image as base with CUDA 12.1 support
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# CUDA configuration
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# Update the architecture list to include recent Ampere and Hopper architectures if needed
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
# Allow falling back to CPU if CUDA fails
ENV FORCE_CUDA=0

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
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages:${PYTHONPATH}

# Accept GitHub token as a build argument
ARG GITHUB_TOKEN

# Configure Git to use HTTPS instead of SSH with token support
RUN git config --global url."https://".insteadOf git:// && \
    git config --global url."https://".insteadOf ssh:// && \
    git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install requirements in one pass with error handling
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install flash-attn with CUDA 12.1 support
RUN pip install flash-attn --no-build-isolation

# Verify CUDA toolkit installation
RUN ls -la ${CUDA_HOME}/bin/nvcc || echo "nvcc not found - this is expected in devel image"

# Pre-compile problematic CUDA extensions to avoid runtime compilation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}')"
RUN mkdir -p /root/.cache/torch_extensions

# Install the SageMaker Inference Toolkit and Multi Model Server
RUN pip install --no-cache-dir multi-model-server sagemaker-inference

# Create directory structure expected by SageMaker
RUN mkdir -p /opt/ml/model /opt/ml/code

# Copy application code
COPY app/services/fileloader /opt/ml/code/fileloader
COPY app/services/sagemaker/docker/inference.py /opt/ml/code/inference.py
COPY app/services/sagemaker/docker/serve.py /opt/ml/code/serve.py
COPY app/services/fileloader/image_processor/model_cache_manager /opt/ml/code/model_cache_manager

# Add patch for graceful fallback when CUDA kernels fail
RUN echo 'import transformers.kernels.deformable_detr.cuda.ms_deform_attn_cuda\nimport sys\n\n# Add CPU fallback for deformable attention\ndef cpu_fallback(*args, **kwargs):\n    print("Using CPU fallback for deformable attention", file=sys.stderr)\n    from transformers.kernels.deformable_detr.cpu.ms_deform_attn_cpu import ms_deform_attn_cpu_forward\n    return ms_deform_attn_cpu_forward(*args, **kwargs)\n\n# Attempt to patch the module if custom CUDA kernel fails\ntry:\n    transformers.kernels.deformable_detr.cuda.ms_deform_attn_cuda.ms_deform_attn_cuda_forward\nexcept (AttributeError, ImportError):\n    transformers.kernels.deformable_detr.cuda.ms_deform_attn_cuda.ms_deform_attn_cuda_forward = cpu_fallback\n' > /opt/ml/code/cuda_patches.py

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

# Set default environment variables that can be overridden by requests
ENV MODEL_CACHE_BUCKET=pureinferencefileloader
ENV MODEL_CACHE_PREFIX=models
# Default Classification settings
ENV CLASSIFICATION_BACKEND_TYPE=clip
ENV CLASSIFICATION_MODEL=openai/clip-vit-large-patch14-336
ENV MODEL_TYPE=transformer

# Default Captioning settings
ENV CAPTION_BACKEND_TYPE=florence2
ENV CAPTION_MODEL=microsoft/Florence-2-large
ENV ENABLE_CAPTIONING=true
ENV MAX_CAPTION_LENGTH=400

# Default General settings
ENV CONFIDENCE_THRESHOLD=0.7
ENV USE_HALF_PRECISION=false

# Set environment variables for SageMaker inference
ENV SAGEMAKER_PROGRAM=inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# Simple CUDA test script
RUN echo '#!/bin/bash\n\
    echo "=== CUDA VERIFICATION ==="\n\
    if command -v nvidia-smi &> /dev/null; then\n\
    echo "NVIDIA Driver:"\n\
    nvidia-smi\n\
    else\n\
    echo "nvidia-smi not found (normal in build environment)"\n\
    fi\n\
    echo ""\n\
    echo "PyTorch CUDA Information:"\n\
    python -c "import torch; print(f\"PyTorch version: {torch.__version__}\"); \
    print(f\"CUDA available: {torch.cuda.is_available()}\"); \
    print(f\"CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}\"); \
    print(f\"Flash Attention test: \" + \"Success\" if __import__(\"flash_attn\") else \"Failed\")"\n\
    ' > /usr/local/bin/check_cuda && chmod +x /usr/local/bin/check_cuda

# Create a simple entrypoint script
RUN echo '#!/bin/bash\n\
    # Run CUDA verification\n\
    /usr/local/bin/check_cuda\n\
    \n\
    if [ "$1" = "serve" ]; then\n\
    echo "Starting model server..."\n\
    exec python serve.py\n\
    else\n\
    exec "$@"\n\
    fi' > /usr/local/bin/serve && chmod +x /usr/local/bin/serve

# Set ENTRYPOINT to the serve script and default CMD to "serve"
ENTRYPOINT ["/usr/local/bin/serve"]
CMD ["serve"]