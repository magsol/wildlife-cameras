# Motion Storage Connection Fix

## Problem

The web UI was correctly detecting motion and displaying it in the "Detection History", but the motion events were not being saved properly:
- No video segments were being saved to the expected storage location
- No pending transfers appeared in the web UI
- No motion detection events were being queued for storage
- The motion_storage backend didn't detect connections from the FastAPI module

## Root Cause

After analyzing the code, I identified the key issue: the motion_storage module was using its own internal instance of `StorageConfig` with default values, while the FastAPI module had its own separate instance of `StorageConfig` with values derived from command-line arguments.

Specifically:

1. In fastapi_mjpeg_server_with_storage.py:
```python
# Set up storage configuration from arguments
storage_config = StorageConfig()
storage_config.local_storage_path = args.storage_path
storage_config.max_disk_usage_mb = args.max_storage
# ... more settings ...
```

2. But when initializing motion_storage:
```python
motion_storage = init_motion_storage(app, camera_config)  # Only passing camera_config!
```

3. Meanwhile, in motion_storage.py:
```python
# Initialize components
storage_config = StorageConfig()  # Creates its OWN instance with default values!
frame_buffer = CircularFrameBuffer(max_size=storage_config.max_ram_segments)
wifi_monitor = WiFiMonitor(storage_config)
transfer_manager = TransferManager(storage_config, wifi_monitor)
motion_recorder = MotionEventRecorder(frame_buffer, storage_config)
```

This resulted in:
- The FastAPI module correctly detecting motion using its configuration
- But the motion_storage module using different storage paths and settings
- Motion events being saved to `/tmp/motion_events` (default) instead of the user-specified location
- The user not seeing any motion events in the web UI's storage section

## The Fix

The solution involves two key changes:

### 1. Pass the storage_config from FastAPI to motion_storage

In fastapi_mjpeg_server_with_storage.py:
```python
# Initialize motion storage
global motion_storage
# Pass both camera_config AND storage_config to the motion_storage module
motion_storage = init_motion_storage(app, camera_config, storage_config)
```

### 2. Update motion_storage to use the provided config

In motion_storage.py, update the initialize function:
```python
def initialize(app=None, camera_config=None, external_storage_config=None):
    """Initialize the motion storage module and integrate with FastAPI server"""
    
    # Reset the shutdown event (in case it was previously set)
    global shutdown_requested
    shutdown_requested.clear()
    
    # Use the provided storage_config if available
    global storage_config, frame_buffer, wifi_monitor, transfer_manager, motion_recorder
    
    # Update our storage config with values from external_storage_config if provided
    if external_storage_config is not None:
        logger.info("Using storage configuration from FastAPI server")
        # Copy all attributes from the external config to our internal config
        for attr in dir(external_storage_config):
            # Skip private/special attributes and methods
            if attr.startswith('_') or callable(getattr(external_storage_config, attr)):
                continue
            # Copy the attribute value if it exists in our storage_config too
            if hasattr(storage_config, attr):
                setattr(storage_config, attr, getattr(external_storage_config, attr))
        logger.info(f"Using storage path: {storage_config.local_storage_path}")
    
    # Reinitialize components with updated config
    frame_buffer = CircularFrameBuffer(max_size=storage_config.max_ram_segments)
    wifi_monitor = WiFiMonitor(storage_config)
    transfer_manager = TransferManager(storage_config, wifi_monitor)
    motion_recorder = MotionEventRecorder(frame_buffer, storage_config)
    
    # Create storage directory if it doesn't exist
    os.makedirs(storage_config.local_storage_path, exist_ok=True)
```

## Benefits

With this fix:
1. The motion_storage module will use the same storage settings as specified in the FastAPI command-line arguments
2. Motion events will be saved to the correct storage location
3. The web UI will show pending transfers and storage statistics properly
4. The entire motion detection and storage pipeline will function correctly

## Technical Details

The root cause of this issue was a common software engineering problem: maintaining synchronized configurations between modules. The FastAPI module and motion_storage module were both using separate instances of the same `StorageConfig` class, but these instances were not synchronized.

By explicitly passing the configuration from one module to the other, we ensure that both modules use the same settings. The motion_storage module still maintains its own instance, but the values are copied from the FastAPI instance during initialization.

This approach is more robust than directly sharing the instance, as it:
1. Keeps module boundaries clean (motion_storage still owns its own configuration)
2. Makes the dependency explicit (the FastAPI module is clearly providing the configuration)
3. Allows for module-specific customization if needed in the future