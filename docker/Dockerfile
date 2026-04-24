FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/backend/harness

# Copy and install Aether harness package
COPY backend/harness/aether /workspace/backend/harness/aether
RUN pip install --upgrade pip \
    && pip install -e /workspace/backend/harness/aether

# Default command keeps container alive for interactive usage
CMD ["tail", "-f", "/dev/null"]
