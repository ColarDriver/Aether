FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Copy and install Aether harness package
COPY pyproject.toml /workspace/pyproject.toml
COPY aether /workspace/aether
RUN pip install --upgrade pip \
    && pip install -e /workspace

# Default command keeps container alive for interactive usage
CMD ["tail", "-f", "/dev/null"]
