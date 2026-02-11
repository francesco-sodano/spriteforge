# How to Use This Template

This document explains how to create a new Python 3.12 project from the **devcontainer-python-template** repository and personalize it for your needs.

---

## Table of Contents

1. [Create a New Repository from the Template](#1-create-a-new-repository-from-the-template)
2. [Repository Structure Overview](#2-repository-structure-overview)
3. [Rename the Project](#3-rename-the-project)
4. [Update Project Metadata](#4-update-project-metadata)
5. [Manage Dependencies](#5-manage-dependencies)
6. [Dev Container Setup](#6-dev-container-setup)
7. [Running Tests, Linting, and Formatting](#7-running-tests-linting-and-formatting)
8. [Docker (Production Image)](#8-docker-production-image)
9. [GitHub Automation](#9-github-automation)
10. [Post-Setup Cleanup](#10-post-setup-cleanup)

---

## 1. Create a New Repository from the Template

1. Go to [**github.com/francesco-sodano/devcontainer-python-template**](https://github.com/francesco-sodano/devcontainer-python-template).
2. Click the green **"Use this template"** button → **"Create a new repository"**.
3. Choose an owner, give the repo a name (e.g., `my-awesome-project`), set visibility, and click **"Create repository"**.
4. Clone the new repository locally:

   ```bash
   git clone https://github.com/<your-user>/my-awesome-project.git
   cd my-awesome-project
   ```

> **Tip:** If you use VS Code with Dev Containers, simply open the cloned repo in VS Code and select **"Reopen in Container"** — the environment will be fully configured automatically.

---

## 2. Repository Structure Overview

```
.
├── .devcontainer/
│   ├── Dockerfile              # Dev container image (Ubuntu 24.04 + uv)
│   ├── devcontainer.json       # Dev container configuration & VS Code settings
│   └── scripts/
│       └── postCreate.sh       # Runs after container creation (installs Python, syncs deps)
├── .github/
│   ├── copilot-instructions.md # GitHub Copilot & AI agent custom instructions
│   ├── dependabot.yml          # Dependabot config (uv, GitHub Actions, devcontainers)
│   ├── ISSUE_TEMPLATE/
│   │   ├── config.yml          # Template chooser config (disables blank issues)
│   │   ├── feature_request.yml # Feature request template (for AI agents)
│   │   └── bug_report.yml      # Bug report template (for AI agents)
│   └── workflows/
│       └── update-lockfile.yml # Auto-updates uv.lock on dependency changes
├── scripts/
│   └── run.sh                  # Helper scripts (empty placeholder)
├── src/
│   └── project_name/           # ← Your package code lives here (rename this)
│       └── __init__.py
├── tests/
│   └── test_app.py             # ← Your tests live here
├── .dockerignore               # Files excluded from Docker builds
├── .gitignore                  # Files excluded from Git
├── Dockerfile                  # Production Docker image
├── LICENSE                     # MIT license
├── pyproject.toml              # Single source of truth for metadata & dependencies
└── README.md                   # Project README (update for your project)
```

### Key Concepts

| Concept | Detail |
|---|---|
| **Language** | Python 3.12 (pinned to `3.12.12` in `postCreate.sh`) |
| **Package manager** | [uv](https://docs.astral.sh/uv/) — fast Python package manager |
| **Build backend** | [hatchling](https://hatch.pypa.io/) |
| **Config file** | `pyproject.toml` is the single source of truth — no `requirements.txt` |
| **Source layout** | `src/` layout (`src/<package_name>/`) |
| **Testing** | pytest |
| **Formatting** | black |
| **Type checking** | mypy |

---

## 3. Rename the Project

Everywhere the placeholder `project_name` appears must be replaced with your actual package name. Use a valid Python identifier: **lowercase, underscores only, no hyphens**.

> **Example:** If your project is called "Data Pipeline", use `data_pipeline` as the package name.

### Step-by-step

#### 3.1 Rename the source package folder

```bash
mv src/project_name src/data_pipeline
```

#### 3.2 Update `pyproject.toml`

Open `pyproject.toml` and update the `[project]` section:

```toml
[project]
name = "data_pipeline"              # ← your package name
version = "0.1.0"
description = "My awesome project." # ← your description
authors = [{ name = "Your Name" }]  # ← your name
```

#### 3.3 Search-and-replace across the entire repo

Run a global find-and-replace for `project_name` → `data_pipeline` across all files. Key files to check:

| File | What to change |
|---|---|
| `pyproject.toml` | `name` field under `[project]` |
| `src/project_name/` | Folder name |
| `tests/test_app.py` | Any future imports (e.g., `from project_name import ...`) |
| `scripts/run.sh` | Any references to the package |
| `README.md` | Project title and description |
| `.github/copilot-instructions.md` | References to `src/project_name/` in the layout description |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | Package name in placeholders and Project Standards block |
| `.github/ISSUE_TEMPLATE/config.yml` | Repository URL and other issue template metadata placeholders |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Package name in placeholders and Project Standards block |

You can use this one-liner to find all occurrences:

```bash
grep -rn "project_name" --include="*.py" --include="*.toml" --include="*.md" --include="*.sh" --include="*.json" --include="*.yml" .
```

#### 3.4 Update the README.md

Replace the template README with your own project description. Delete the "How to customize this template" instructions section.

---

## 4. Update Project Metadata

All project metadata lives in `pyproject.toml`:

```toml
[project]
name = "data_pipeline"
version = "0.1.0"
description = "A brief description of what your project does."
readme = "README.md"
requires-python = ">=3.12.12,<3.13"
authors = [{ name = "Your Name" }]
license = { text = "MIT" }
dependencies = []
```

Fields you should personalize:

| Field | Purpose |
|---|---|
| `name` | Your Python package name (must match the folder under `src/`) |
| `version` | Semantic version — start at `0.1.0` |
| `description` | One-line summary of the project |
| `authors` | Your name and optionally email: `{ name = "You", email = "you@example.com" }` |
| `license` | Change if not MIT — also update the `LICENSE` file |
| `requires-python` | Python version constraint (default: `>=3.12.12,<3.13`) |

---

## 5. Manage Dependencies

This template uses **uv** as the package manager with `pyproject.toml` as the single source of truth. **No `requirements.txt` files are used.**

### 5.1 Add a runtime dependency

Runtime dependencies are packages your project needs to function (e.g., `requests`, `fastapi`, `pandas`).

```bash
# Add a single package
uv add requests

# Add with a version constraint
uv add "requests>=2.31,<3"

# Add multiple packages at once
uv add requests fastapi sqlalchemy
```

This will update the `dependencies` list in `pyproject.toml`:

```toml
[project]
dependencies = [
    "requests>=2.31,<3",
    "fastapi>=0.115",
    "sqlalchemy>=2.0",
]
```

### 5.2 Add a dev dependency

Dev dependencies are tools used only during development (testing, linting, formatting) and are **not** shipped with your package. They live in the `[dependency-groups]` section.

```bash
# Add to the dev group
uv add --group dev ruff

# Add with a version pin
uv add --group dev "ruff==0.11.12"
```

This updates the `[dependency-groups]` section:

```toml
[dependency-groups]
dev = [
    "black==26.1.0",
    "mypy==1.19.1",
    "pytest==9.0.2",
    "ruff==0.11.12",   # ← newly added
]
```

### 5.3 Remove a dependency

```bash
# Remove a runtime dependency
uv remove requests

# Remove a dev dependency
uv remove --group dev ruff
```

### 5.4 Sync your environment

After manually editing `pyproject.toml` or pulling changes, sync your virtualenv:

```bash
uv sync --group dev
```

> **Note:** The dev container's `postCreate.sh` script runs `uv sync --group dev` automatically whenever the container is (re)built, so your environment is always up to date on container start.

### 5.5 Update dependencies to latest versions

```bash
# Update all dependencies
uv lock --upgrade
uv sync --group dev

# Update a specific package
uv lock --upgrade-package requests
uv sync --group dev
```

### 5.6 The lock file (`uv.lock`)

`uv` generates a `uv.lock` file that pins exact resolved versions. **Commit this file** to version control for reproducible builds. The included GitHub Actions workflow (`.github/workflows/update-lockfile.yml`) will automatically regenerate `uv.lock` when Dependabot updates `pyproject.toml`.

---

## 6. Dev Container Setup

The template comes with a fully configured [Dev Container](https://containers.dev/) for VS Code.

### What's included

| Component | Description |
|---|---|
| **Base image** | `mcr.microsoft.com/devcontainers/base:ubuntu24.04` |
| **uv** | Installed from the `ghcr.io/astral-sh/uv:latest` image |
| **Python 3.12** | Installed via `uv python install` in `postCreate.sh` |
| **Virtual environment** | Created at `.venv/` in the workspace root |
| **Dev features** | Git, Azure CLI (with Bicep), Node.js (LTS), Docker-in-Docker, Azure Developer CLI, GitHub Copilot CLI, PowerShell |

### Pre-installed VS Code extensions

- GitHub Copilot & Copilot Chat
- Python, Pylance, Debugpy
- Jupyter suite
- Docker
- Azure tools (Resource Groups, CLI, MCP Server)
- PowerShell

### VS Code settings (auto-configured)

- Line endings: `LF`
- Format on save: enabled
- Python formatter: `black`
- Linting: `mypy` enabled
- Testing: `pytest` enabled
- **Copilot agent auto-approve:** enabled (`chat.tools.global.autoApprove: true`) — agent tools run without manual confirmation. Set to `false` in `devcontainer.json` if you prefer to approve each tool invocation.
- **Copilot agent max requests:** `100` per session (`chat.agent.maxRequests`) — controls how many requests the agent can make in a single session. Lower this value for tighter control.

### How to use

1. Open the repo in VS Code.
2. When prompted, click **"Reopen in Container"** (or run the command `Dev Containers: Reopen in Container`).
3. Wait for the container to build and `postCreate.sh` to finish.
4. The `.venv` is auto-activated in every new terminal session.

### Customizing the Dev Container

- **Add system packages:** Edit `.devcontainer/Dockerfile` and add `apt-get install` commands.
- **Add Dev Container Features:** Edit `.devcontainer/devcontainer.json` under `"features"`.
- **Add VS Code extensions:** Edit `.devcontainer/devcontainer.json` under `"customizations" > "vscode" > "extensions"` — add the extension ID.
- **Change post-create behavior:** Edit `.devcontainer/scripts/postCreate.sh`.
- **Forward ports:** Uncomment and populate `"forwardPorts"` in `devcontainer.json`.

---

## 7. Running Tests, Linting, and Formatting

All dev tools are pre-installed in the `dev` dependency group.

### Run tests

```bash
pytest
```

Tests live in the `tests/` directory. The template includes an empty `tests/test_app.py` — add your tests there or create new test files following the `test_*.py` naming convention.

### Format code

```bash
black .
```

### Type checking

```bash
mypy src/
```

### Run all checks at once

```bash
black . && mypy src/ && pytest
```

---

## 8. Docker (Production Image)

The root `Dockerfile` is a production-ready image definition, separate from the Dev Container Dockerfile (`.devcontainer/Dockerfile`). It already copies your source code, installs only runtime dependencies (excluding the `dev` group), and sets a default entry point.

```dockerfile
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
```

### Customizing for your project

After renaming `project_name`, update the `CMD` line to match your package name:

```dockerfile
CMD ["python", "-m", "data_pipeline"]
```

### Key design choices

- **Layer caching:** `pyproject.toml` and `uv.lock` are copied before the source code so that dependency installation is cached and only re-runs when dependencies change.
- **No dev tools in production:** `uv sync --no-dev` excludes `black`, `mypy`, `pytest`, and any other packages in the `dev` dependency group.
- **`.dockerignore`:** Pre-configured to exclude `.venv/`, `__pycache__/`, `.git/`, and other unnecessary files from the build context.

### Build and run

```bash
docker build -t my-app .
docker run --rm my-app
```

---

## 9. GitHub Automation

### Dependabot

The template includes a `.github/dependabot.yml` that automatically checks for updates every Sunday for:

| Ecosystem | What it tracks |
|---|---|
| **uv** | Python dependencies in `pyproject.toml` / `uv.lock` |
| **github-actions** | Action versions in `.github/workflows/` |
| **devcontainers** | Dev Container feature versions in `.devcontainer/` |

### Lock file auto-update workflow

`.github/workflows/update-lockfile.yml` runs automatically when Dependabot pushes changes to `pyproject.toml` on a `dependabot/**` branch. It regenerates `uv.lock` and commits the result so the PR stays consistent.

### Copilot instructions

`.github/copilot-instructions.md` provides project-aware context to GitHub Copilot and AI agents. Update the layout references and project name after renaming.

### Issue Templates

The template includes structured GitHub Issue Templates in `.github/ISSUE_TEMPLATE/`. They are designed to work with the AI agent workflow, which is documented in detail in `.github/copilot-instructions.md` (the canonical source for the autonomous workflow and process).

| Template | File | Purpose |
|---|---|---|
| **Feature Request** | `.github/ISSUE_TEMPLATE/feature_request.yml` | Propose new features/enhancements with priority, complexity, implementation details, related files, breaking changes flag, acceptance criteria, and test plans |
| **Bug Report** | `.github/ISSUE_TEMPLATE/bug_report.yml` | Report bugs with priority, complexity, reproduction steps, current workaround, root cause analysis, proposed fixes, and regression test plans |

For specifics on how AI agents should interpret and execute these templates (including priorities, complexity, project standards, and dependencies), refer to `.github/copilot-instructions.md`.
- **Acceptance Criteria** with standard quality gates pre-defined
- **Missing information rule** — the agent must ask for clarification instead of guessing

**Feature Request** additionally includes:
- **Related Files & Code** — points the agent to existing files to read or extend
- **Breaking Changes flag** — tells the agent whether backward compatibility must be preserved
- **Out of Scope** — explicitly prevents the agent from doing unnecessary work

**Bug Report** additionally includes:
- **Current Workaround** — warns the agent about temporary fixes to preserve or remove
- **Root Cause Analysis** — saves the agent diagnostic time
- **Affected Code & Location** — directs the agent to the exact file/function/line

**How it works:**

1. A human creates an issue using one of the templates, filling in all required sections.
2. An AI agent (e.g., GitHub Copilot) reads the issue, applies the project standards, and checks for dependencies on other issues before starting.
3. The agent follows the canonical workflow defined in `.github/copilot-instructions.md` to implement the solution.

Keep the templates in sync with `.github/copilot-instructions.md` — that file is the source of truth for agent standards and workflow.

---

## 10. Post-Setup Cleanup

After personalizing the template, tidy up:

- [ ] Rename `src/project_name/` to your package name.
- [ ] Update all fields in `pyproject.toml` (name, description, authors, license).
- [ ] Search-and-replace `project_name` across the entire repo.
- [ ] Rewrite `README.md` with your project's own documentation.
- [ ] Update `.github/copilot-instructions.md` with your new package name and layout.
- [ ] Update `.github/ISSUE_TEMPLATE/config.yml` — replace `REPLACE_WITH_OWNER/REPLACE_WITH_REPO` with your actual GitHub owner and repo name.
- [ ] Update the **Project Standards** block in `.github/ISSUE_TEMPLATE/feature_request.yml` and `bug_report.yml` if your standards differ.
- [ ] Update the `LICENSE` file if you want a different license or author.
- [ ] Delete this `README-template.md` file — it's no longer needed.
- [ ] Add your initial source code to `src/<your_package>/`.
- [ ] Add your first tests to `tests/`.
- [ ] Commit everything and push.

---

## Quick Reference

| Task | Command |
|---|---|
| Add runtime dependency | `uv add <package>` |
| Add dev dependency | `uv add --group dev <package>` |
| Remove dependency | `uv remove <package>` |
| Sync environment | `uv sync --group dev` |
| Run tests | `pytest` |
| Format code | `black .` |
| Type check | `mypy src/` |
| Update all deps | `uv lock --upgrade && uv sync --group dev` |
| Build package | `uv build` |
| Find all `project_name` refs | `grep -rn "project_name" .` |
