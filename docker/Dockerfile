FROM node:22-bookworm-slim AS node

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/aether/.venv \
    VIRTUAL_ENV=/opt/aether/.venv \
    PATH=/opt/aether/.venv/bin:/usr/local/bin:${PATH}

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules

RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

FROM base AS source

COPY pyproject.toml uv.lock uv.toml ./
COPY aether ./aether
COPY tui/package.json tui/package-lock.json ./tui/

FROM source AS runtime

RUN uv sync --frozen --no-dev
RUN cd tui && npm ci
COPY tui ./tui
RUN cd tui \
    && npm run build \
    && npm prune --omit=dev \
    && npm cache clean --force

ENV NODE_ENV=production

CMD ["aether"]

FROM source AS dev

ENV NODE_ENV=development

RUN uv sync --frozen
RUN cd tui && npm ci
COPY tui ./tui
RUN cd tui && npm run build

CMD ["tail", "-f", "/dev/null"]
