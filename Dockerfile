FROM mcr.microsoft.com/devcontainers/base:ubuntu24.04

# Install uv, the project's package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Ensure python3 is the default python
RUN apt-get update -y \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get install -y -q --no-install-recommends python-is-python3

# Set working directory
WORKDIR /app

# Copy dependency metadata first for layer caching
COPY pyproject.toml uv.lock ./

# Copy source code
COPY src/ ./src/

# Install only runtime dependencies (no dev group)
RUN uv venv .venv && . .venv/bin/activate && uv sync --no-dev
ENV PATH="/app/.venv/bin:$PATH"

# Default entrypoint — replace project_name with your package name
CMD ["python", "-m", "project_name"]
