# Configuration Refactoring Plan

## Overview

This document outlines the refactoring plan to centralize configuration management across all wildlife camera system components.

## Current State (Problems)

1. **Scattered Configuration**: Settings spread across 4 modules
   - `fastapi_mjpeg_server_with_storage.py` - CameraConfig class
   - `motion_storage.py` - StorageConfig class
   - `optical_flow_analyzer.py` - Parameters passed to constructors
   - `storage_server.py` - Hardcoded or argument-based config

2. **No Single Source of Truth**: Each module defines its own defaults

3. **Limited Configuration Methods**:
   - Some command-line args in server (incomplete)
   - No environment variable support
   - No configuration file support
   - Changes require code edits

4. **No Configuration Validation**: No centralized validation of settings

## New Architecture

### Components Created

1. **`config.py`** - Centralized configuration module
   - Single dataclass hierarchy for all settings
   - YAML file loader
   - Environment variable overrides
   - Configuration validation
   - Default value management

2. **`config.yaml`** - Human-editable configuration file
   - All settings in one place with comments
   - Standard location search (./config.yaml, ~/.wildlife-camera/config.yaml, /etc/wildlife-camera/config.yaml)

3. **`cli.py`** - Unified command-line interface
   - Arguments for all common settings
   - Overrides config file and environment variables

### Configuration Priority (Highest to Lowest)

1. **Command-line arguments** (--width 1920)
2. **Environment variables** (WC_CAMERA_WIDTH=1920)
3. **Config file** (config.yaml)
4. **Defaults** (defined in config.py)

### Configuration Sections

```
WildlifeCameraConfig
├── camera: CameraConfig
│   └── Resolution, frame rate, display settings
├── motion_detection: MotionDetectionConfig
│   └── Threshold, sensitivity, history
├── optical_flow: OpticalFlowConfig
│   └── Feature detection, grid, performance
├── storage: StorageConfig
│   └── Local/remote paths, throttling, WiFi
├── optical_flow_storage: OpticalFlowStorageConfig
│   └── Pattern database, signatures
├── server: ServerConfig
│   └── Host, port, logging
└── storage_server: StorageServerConfig
    └── Upload server configuration
```

## Refactoring Tasks

### 1. FastAPI Server (`fastapi_mjpeg_server_with_storage.py`)

**Changes:**
- Remove `CameraConfig` class definition
- Import from `config` module instead
- Update `lifespan()` to load config using `cli.load_config_with_cli()`
- Pass config sections to other modules
- Update argument parser (or remove in favor of cli.py)

**Example:**
```python
# OLD
camera_config = CameraConfig()

# NEW
from config import get_config
from cli import load_config_with_cli

config, args = load_config_with_cli()
camera_config = config.camera
```

### 2. Motion Storage (`motion_storage.py`)

**Changes:**
- Remove `StorageConfig` class definition
- Accept config object in `initialize()` function
- Update all references to use passed config

**Example:**
```python
# OLD
def initialize(local_path: str = "/tmp/motion_events", ...):
    config = StorageConfig()
    config.local_storage_path = local_path

# NEW
def initialize(storage_config: StorageConfig, optical_flow_config: OpticalFlowStorageConfig):
    # Use passed configuration objects
```

### 3. Optical Flow Analyzer (`optical_flow_analyzer.py`)

**Changes:**
- Update `OpticalFlowAnalyzer.__init__()` to accept config object
- Replace individual parameters with config fields
- Update `MotionPatternDatabase.__init__()` similarly

**Example:**
```python
# OLD
def __init__(self, feature_max=100, quality_level=0.3, ...):
    self.feature_max = feature_max

# NEW
def __init__(self, config: OpticalFlowConfig):
    self.config = config
    self.feature_max = config.feature_max
```

### 4. Storage Server (`storage_server.py`)

**Changes:**
- Add config loading at startup
- Remove/update argument parser to use cli.py
- Update all hardcoded values to use config

**Example:**
```python
# OLD
app = FastAPI()
STORAGE_PATH = "/tmp/received_events"

# NEW
from config import get_config
config = get_config()
app = FastAPI()
STORAGE_PATH = config.storage_server.storage_path
```

## Migration Path

### Option A: Gradual Migration (Recommended)

1. **Phase 1**: Add config system alongside existing code
   - Keep existing CameraConfig/StorageConfig classes
   - Add config.py and config.yaml
   - Update startup to optionally use new config
   - Both systems work in parallel

2. **Phase 2**: Migrate modules one by one
   - Start with storage_server (simplest)
   - Then fastapi_mjpeg_server
   - Then motion_storage
   - Finally optical_flow_analyzer

3. **Phase 3**: Remove old config classes
   - Delete old @dataclass definitions
   - Remove old argument parsers
   - Clean up imports

### Option B: Big Bang Migration

- Refactor all 4 modules at once
- Higher risk but faster completion
- Requires comprehensive testing afterward

## Usage Examples

### Basic Usage (Default Configuration)

```bash
# Uses config.yaml if present, otherwise defaults
pixi run start
```

### With Custom Config File

```bash
pixi run start --config /path/to/my-config.yaml
```

### With Environment Variables

```bash
export WC_CAMERA_WIDTH=1920
export WC_CAMERA_HEIGHT=1080
export WC_STORAGE_PATH=/mnt/external/motion_events
pixi run start
```

### With Command-Line Overrides

```bash
pixi run start --width 1920 --height 1080 --frame-rate 15 --disable-uploads
```

### Generate Default Config

```bash
python cli.py --generate-config
# Creates config.yaml with all options and documentation
```

## Benefits

1. **Single Source of Truth**: All settings in one place
2. **Easy Deployment**: Edit config.yaml instead of code
3. **Environment Flexibility**: Use env vars in Docker/systemd
4. **Better Documentation**: Config file is self-documenting
5. **Validation**: Catch invalid settings at startup
6. **Override Flexibility**: Choose config method based on deployment
7. **Version Control**: Track configuration changes separately from code

## Backward Compatibility

The refactored system will maintain backward compatibility by:

1. **Default values remain the same**: Existing behavior unchanged
2. **Optional config file**: System works without config.yaml
3. **Command-line args**: Can still use args for common settings

## Testing Plan

After refactoring:

1. **Unit tests**: Test config loader with various inputs
2. **Integration tests**: Test full system with different config methods
3. **Migration test**: Verify old behavior matches new behavior with defaults
4. **Environment test**: Test on Raspberry Pi with actual hardware

## Rollback Plan

If issues arise:
1. Git revert to pre-refactor commit
2. All existing code still functional as-is
3. Can complete refactor later with lessons learned

## Recommendation

**Start with Option A (Gradual Migration)** because:
- Lower risk
- Can test each module independently
- Can roll back individual changes
- Users can adopt at their own pace
- Easier to debug issues

## Next Steps

1. Review this plan
2. Choose migration approach (A or B)
3. Decide which module to start with
4. Execute refactoring
5. Update documentation
6. Test on Raspberry Pi
