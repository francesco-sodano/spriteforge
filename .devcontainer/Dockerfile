FROM mcr.microsoft.com/devcontainers/base:ubuntu24.04

# Install uv, the project's package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Ensure python3 is the default python
RUN apt-get update -y \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get install -y -q --no-install-recommends python-is-python3

# Dev dependencies are installed in postCreate.sh inside a workspace venv
