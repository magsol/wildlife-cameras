# Resolution Settings Guide

## Overview

The wildlife camera framework uses **three different resolution settings** for different purposes. Understanding which setting controls what is crucial for configuring your system correctly.

## The Three Resolution Settings

### 1. Camera Resolution (Live Feed)

**Config Location**: `camera.width` and `camera.height`

**Purpose**: Controls the resolution of the live video feed shown in the web UI and recorded to disk.

**What it affects**:
- Video stream quality in web browser
- Recording quality of motion events
- CPU load (higher resolution = more processing)
- Network bandwidth (higher resolution = more data)

**Typical Values**:
```yaml
camera:
  # For 1080p (high quality, high CPU):
  width: 1920
  height: 1080

  # For 720p (balanced):
  width: 1280
  height: 720

  # For performance (low CPU):
  width: 640
  height: 480
```

**When to change**:
- You want higher quality video in the web UI
- You want better quality recordings of motion events
- You have sufficient CPU/network bandwidth

---

### 2. Optical Flow Processing Resolution

**Config Location**: `optical_flow.max_resolution_width` and `optical_flow.max_resolution_height`

**Purpose**: Downscales frames for optical flow analysis to reduce CPU load.

**What it affects**:
- CPU usage during motion pattern analysis
- Speed of optical flow feature detection
- Does NOT affect live feed quality or recordings

**Typical Values**:
```yaml
optical_flow:
  # Low CPU usage (recommended):
  max_resolution_width: 320
  max_resolution_height: 240

  # Very low CPU usage (Raspberry Pi Zero):
  max_resolution_width: 240
  max_resolution_height: 180

  # Higher quality analysis (more CPU):
  max_resolution_width: 640
  max_resolution_height: 480
```

**When to change**:
- CPU usage is too high (reduce resolution)
- Optical flow classification accuracy is poor (increase resolution)
- Running on low-power device like Pi Zero (reduce to 240x180)

**Important**: This resolution is completely independent of the camera resolution. Optical flow analysis always works on a downscaled copy of the frame for performance.

---

### 3. Thumbnail Resolution

**Config Location**: `storage.thumbnail_width` and `storage.thumbnail_height`

**Purpose**: Size of preview thumbnail images saved with motion events.

**What it affects**:
- Size of thumbnail JPEG files on disk
- Quick preview quality in motion event list
- Does NOT affect live feed or main recordings

**Typical Values**:
```yaml
storage:
  # Standard thumbnails:
  thumbnail_width: 320
  thumbnail_height: 240

  # Smaller thumbnails (less disk space):
  thumbnail_width: 160
  thumbnail_height: 120

  # Larger thumbnails (better preview):
  thumbnail_width: 640
  thumbnail_height: 480
```

**When to change**:
- You want better quality preview images
- You want to save disk space (reduce size)
- You're generating many thumbnails per event

---

## Configuration Examples

### High Quality Setup (Raspberry Pi 4, 4GB RAM)
```yaml
camera:
  width: 1920          # 1080p live feed
  height: 1080
  frame_rate: 30

optical_flow:
  max_resolution_width: 320   # Still downscaled for CPU efficiency
  max_resolution_height: 240

storage:
  thumbnail_width: 640        # Larger thumbnails for better previews
  thumbnail_height: 480
```

### Balanced Setup (Raspberry Pi 4, 2GB RAM)
```yaml
camera:
  width: 1280          # 720p live feed
  height: 720
  frame_rate: 20

optical_flow:
  max_resolution_width: 320   # Standard optical flow resolution
  max_resolution_height: 240

storage:
  thumbnail_width: 320        # Standard thumbnails
  thumbnail_height: 240
```

### Performance Setup (Raspberry Pi Zero 2W)
```yaml
camera:
  width: 640           # Low resolution live feed
  height: 480
  frame_rate: 10

optical_flow:
  enabled: false       # Disable optical flow to save CPU
  # OR use very low resolution:
  max_resolution_width: 240
  max_resolution_height: 180

storage:
  thumbnail_width: 160        # Small thumbnails
  thumbnail_height: 120
  generate_thumbnails: false  # Or disable entirely
```

---

## Common Issues

### Issue: "I set camera.width to 1920 but web UI still shows low resolution"

**Solution**: Check that your config.yaml changes were saved and the server was restarted:
```bash
# Verify config file
grep -A 2 "camera:" config.yaml

# Restart server
sudo systemctl restart wildlife-camera
# Or if running manually:
pixi run start
```

### Issue: "High CPU usage after increasing camera resolution"

**Explanation**: Higher camera resolution means more pixels to process for:
- Motion detection
- MJPEG encoding
- Network streaming
- Optical flow (if enabled)

**Solutions**:
1. Reduce frame rate: `frame_rate: 15` (instead of 30)
2. Ensure optical flow uses low resolution: `max_resolution_width: 240`
3. Increase frame skip: `frame_skip: 4` (process every 4th frame)
4. Disable expensive features: `optical_flow.visualization: false`

### Issue: "Optical flow classification not accurate"

**Solution**: The optical flow processing resolution might be too low. Try:
```yaml
optical_flow:
  max_resolution_width: 480
  max_resolution_height: 360
  frame_skip: 2
```

Note: This will increase CPU usage. Monitor with `htop`.

---

## Removed Settings

### `optical_flow.max_resolution` (DEPRECATED)

**Old format** (tuple):
```yaml
optical_flow:
  max_resolution: [320, 240]  # ❌ No longer used
```

**New format** (separate width/height):
```yaml
optical_flow:
  max_resolution_width: 320   # ✅ Use this
  max_resolution_height: 240
```

**Reason for change**: Consistency with other resolution settings (camera, thumbnail) which use separate width/height parameters.

### `storage.optical_flow_max_resolution` (REMOVED)

This was a duplicate of `optical_flow.max_resolution` and has been removed to eliminate confusion.

---

## Summary Table

| Setting | Config Path | Purpose | Affects Live Feed? | Affects CPU? |
|---------|------------|---------|-------------------|--------------|
| **Live Feed** | `camera.width/height` | Web UI video quality | ✅ Yes | High impact |
| **Optical Flow** | `optical_flow.max_resolution_width/height` | Motion analysis performance | ❌ No | Medium impact |
| **Thumbnails** | `storage.thumbnail_width/height` | Preview image size | ❌ No | Low impact |

---

## Resolution and Performance

### CPU Impact by Camera Resolution

| Resolution | Pixels | Relative CPU Load | Use Case |
|------------|--------|-------------------|----------|
| 640×480 | 307K | 1× (baseline) | Performance, Pi Zero 2W |
| 1280×720 | 922K | ~3× | Balanced quality, Pi 4 2GB |
| 1920×1080 | 2073K | ~6-7× | High quality, Pi 4 4GB+ |

**Note**: Optical flow processing resolution is independent - it can stay at 320×240 even with 1080p camera feed.

### Memory Usage

- **640×480**: ~1-2 MB per frame
- **1280×720**: ~3-4 MB per frame
- **1920×1080**: ~6-8 MB per frame

With RAM buffer of 30 seconds @ 30fps:
- **640×480**: ~60-120 MB
- **1280×720**: ~180-240 MB
- **1920×1080**: ~360-480 MB

Adjust `storage.ram_buffer_seconds` if memory constrained.

---

## Best Practices

1. **Start with defaults** (640×480) and increase if needed
2. **Monitor CPU usage** with `htop` after changes
3. **Keep optical flow resolution low** (320×240 or lower)
4. **Test thoroughly** after resolution changes
5. **Document your settings** for future reference

---

**Last Updated**: 2025-10-14
**Framework Version**: 1.0.0
