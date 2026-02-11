# project_name

Brief description of what this project does.

## Features

- Feature 1
- Feature 2
- Feature 3

## Requirements

- Python 3.12

## Installation

```bash
git clone https://github.com/<your-user>/project_name.git
cd project_name
uv sync --group dev
```

## Usage

```bash
python -m project_name
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) as the package manager and a [Dev Container](https://containers.dev/) for a consistent development environment.

### Quick start

1. Open the repo in VS Code and select **"Reopen in Container"**.
2. The dev container installs Python 3.12, creates a `.venv`, and syncs all dependencies automatically.

### Project structure

```
src/project_name/   # Package source code
tests/              # Tests (pytest)
scripts/            # Helper scripts
```

### Run tests

```bash
pytest
```

### Format & lint

```bash
black .
mypy src/
```

## Docker

```bash
docker build -t project_name .
docker run --rm project_name
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.