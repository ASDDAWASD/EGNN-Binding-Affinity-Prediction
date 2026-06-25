# GPU-capable image for the MadraX-based ddG synthetic dataset pipeline.
# Build:  docker build -t foldx-ddg-pipeline .
# Run:    docker run --rm -it --gpus all -v "$(pwd)/artifacts:/app/artifacts" foldx-ddg-pipeline
# (drop --gpus all if no NVIDIA GPU / nvidia-container-toolkit is available;
#  the scripts fall back to CPU automatically via torch.cuda.is_available())
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir uv

WORKDIR /app

# Install dependencies first so this layer is cached unless the manifest changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Now bring in the actual source/data.
COPY synthetic_ppi_dataset.py generate_tcr_pmhc_dataset.py README.md ./
COPY stcrdab_structures ./stcrdab_structures

CMD ["/bin/bash"]
