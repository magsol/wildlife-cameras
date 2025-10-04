# Debugging Motion Detection with Enhanced Logging

I've added extensive logging throughout the codebase to help diagnose why motion events are detected but not properly saved. The focus is particularly on tracking motion events that might be too brief to trigger recording.

## Key Log Tags to Look For

I've added special log tags to make it easier to track the complete flow of a motion event:

### 1. Motion Detection Flow

- `[MOTION_DETECTED]` - When motion is first detected in the FastAPI server
- `[EVENT_STARTED]` - When recording of a motion event begins
- `[EVENT_DURATION]` - Shows the duration check when motion stops
- `[EVENT_REJECTED]` - When a motion event is rejected (e.g., too short)
- `[EVENT_ACCEPTED]` - When a motion event is accepted for processing

### 2. Critical Debugging 

- `[CRITICAL_DEBUG]` - Detailed low-level debug information
- `[MOTION_FLOW]` - General motion detection flow information
- `[TRANSFER_FLOW]` - Information about event transfers

## Testing Procedure

1. **Start the server with logging enabled**:
   ```
   PYTHONUNBUFFERED=1 python fastapi_mjpeg_server_with_storage.py --storage-path /path/to/test/storage > debug_log.txt 2>&1
   ```

2. **Create brief motion events**:
   - Move briefly in front of the camera (1-2 seconds)
   - This should trigger `[MOTION_DETECTED]` logs

3. **Create longer motion events**:
   - Move continuously in front of the camera for 5+ seconds
   - This should create events long enough to pass the minimum duration threshold

4. **Check the logs**:
   - Look for complete sequences: `[MOTION_DETECTED]` -> `[EVENT_STARTED]` -> `[EVENT_DURATION]` -> either `[EVENT_ACCEPTED]` or `[EVENT_REJECTED]`
   - If events are being rejected, check the duration against `min_motion_duration_sec`

## What to Look For

### 1. Brief Motion Events

If motion events are being detected but not saved, you should see:

```
[MOTION_DETECTED] Motion detected! Regions: 2
[EVENT_STARTED] NEW MOTION EVENT STARTED - ID: motion_20250923_164201_123
[EVENT_DURATION] Motion event motion_20250923_164201_123: duration=1.542s, min=3.0s, frames=32
[EVENT_REJECTED] Motion event motion_20250923_164201_123 REJECTED - duration too short
```

This would confirm that events are being properly detected but filtered out due to the duration threshold.

### 2. Connection Issues

If there are issues with the connection between motion detection and recording:

```
[MOTION_DETECTED] Motion detected! Regions: 2
[CRITICAL_DEBUG] CircularFrameBuffer.add_frame called
[CRITICAL_DEBUG] ERROR: stream_buffer.raw_frame is None
```

This would indicate frames aren't being properly passed between components.

### 3. Storage Path Issues

If there are issues with the storage path:

```
[EVENT_ACCEPTED] Motion event will be saved to /tmp/motion_events/motion_20250923_164201_123
[CRITICAL_DEBUG] ERROR: Failed to create directory for event!
```

### 4. Minimum Motion Duration

Pay particular attention to the motion event duration compared to the minimum threshold:

```
[EVENT_DURATION] Motion event: duration=1.542s, min=3.0s, frames=32
```

The default `min_motion_duration_sec` is 3.0 seconds. If you're consistently seeing events just under this threshold, you might want to try lowering it temporarily for testing.

## Modifying the Duration Threshold

To test if brief motions are the issue, you can modify the `min_motion_duration_sec` setting in the `StorageConfig` class:

1. In `motion_storage.py`, locate the `StorageConfig` class
2. Change `min_motion_duration_sec: int = 3` to a lower value like `min_motion_duration_sec: int = 1`
3. Restart the server and test again

## Next Steps

If the logs confirm that motion events are being detected but rejected due to the duration threshold:

1. Consider if the threshold should be lowered permanently
2. Check if the camera's frame rate and motion detection parameters are appropriate
3. Make sure the CircularFrameBuffer is properly receiving frames (look for `CircularFrameBuffer size` logs)

If the logs show a different issue:

1. Look for specific error messages in the `[CRITICAL_DEBUG]` logs
2. Check for file system or permission issues
3. Verify all required dependencies are installed