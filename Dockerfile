# Builder stage: Use devel image for building (especially flash_attn needs nvcc)
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    curl \
    ca-certificates \
    git \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files for uv sync
COPY pyproject.toml uv.lock* ./

# Install Python packages with uv (including flash_attn which needs nvcc)
RUN uv sync --frozen || uv sync
RUN uv pip install flash_attn==2.7.4.post1 --no-build-isolation

# Runtime stage: Use runtime image (lighter, no build tools)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set working directory for project mount
WORKDIR /app

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["/bin/bash"]
