# Development Guide for Wildlife Camera System

This document explains how to set up your development environment for the Raspberry Pi Wildlife Camera project.

## Environment Setup

This project uses [pixi](https://pixi.sh/) for environment management, which provides isolated environments similar to conda.

### Prerequisites

- On Raspberry Pi: 
  - Python 3.9+
  - System-installed `picamera2` package
- On any system:
  - Python 3.9+
  - `pixi` package manager

### Setup Instructions

#### Automated Setup

1. Run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Check if you're on a Raspberry Pi
- Install pixi if needed
- Install system dependencies on Raspberry Pi
- Create a pixi environment with all required packages

#### Manual Setup

1. Install pixi if you don't have it:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

2. On Raspberry Pi, install system dependencies:

```bash
sudo apt update
sudo apt install -y python3-picamera2 --no-install-recommends
sudo apt install -y ffmpeg libopencv-dev
```

3. Create the pixi environment:

```bash
pixi install --feature dev,lint
```

### Using the Environment

1. Activate the environment:

```bash
pixi shell
```

2. The environment is configured with `system-site-packages=true` to access the system-installed `picamera2` package.

## Development Workflow

### Running the Application

```bash
# Start the main camera server
pixi run start

# Start just the storage server
pixi run storage-server
```

### Testing

```bash
# Run all tests
pixi run test

# Run tests with verbose output
pixi run test-verbose

# Run tests with coverage report
pixi run test-coverage
```

### Code Quality

```bash
# Format code
pixi run format

# Run linter
pixi run lint

# Run linter with auto-fix
pixi run check

# Sort imports
pixi run sort-imports

# Run full development workflow (lint, format, test)
pixi run dev-workflow
```

## Environment Details

The project uses three environments:

1. **default** - Base environment with core dependencies
   - fastapi, uvicorn, opencv, numpy, pydantic, requests

2. **dev** - For development and testing
   - All base dependencies plus pytest and related testing tools

3. **lint** - For code quality
   - ruff, black, isort for formatting and linting

All environments are configured to access system site packages to use the system-installed `picamera2` library.

## Known Issues

- The `picamera2` package must be installed at the system level using apt, as it's not available via pip or conda.
- When running on non-Raspberry Pi systems, the code will detect the absence of Pi-specific hardware and disable those features.