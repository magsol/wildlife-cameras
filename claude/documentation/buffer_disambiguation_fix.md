# Frame Buffer Disambiguation Fix

## Problem

After implementing the initial threading fix for the PiCamera2 integration, a new error appeared:

```
Exception in thread Thread-6 (thread_poll):
Traceback (most recent call last):
  File "/usr/lib/python3.11/threading.py", line 1038, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.11/threading.py", line 975, in run
    self._target(*self._args, **self._kwargs)
  File "/usr/lib/python3/dist-packages/picamera2/encoders/v4l2_encoder.py", line 257, in thread_poll
    self.outputframe(b, keyframe, (buf.timestamp.tv_sec * 1000000) + buf.timestamp.tv_usec)
  File "/usr/lib/python3/dist-packages/picamera2/encoders/encoder.py", line 340, in outputframe
    out.outputframe(frame, keyframe, timestamp, packet, audio)
  File "/usr/lib/python3/dist-packages/picamera2/outputs/fileoutput.py", line 93, in outputframe
    self._write(frame, timestamp)
  File "/usr/lib/python3/dist-packages/picamera2/outputs/fileoutput.py", line 123, in _write
    self._fileoutput.write(frame)
  File "wildlife-cameras/claude/motion_storage.py", line 1215, in write_wrapper
    result = frame_buffer._original_write(buf)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'CircularFrameBuffer' object has no attribute '_original_write'
```

## Root Cause

The codebase had two nearly-identical classes acting as frame buffers, causing confusion:

1. `FrameBuffer` in `fastapi_mjpeg_server_with_storage.py`: This is the actual buffer that PiCamera2 uses to write frames. It inherits from `io.BufferedIOBase` and provides a `write()` method compatible with PiCamera2's `FileOutput` class.

2. `CircularFrameBuffer` in `motion_storage.py`: This is a separate buffer used for storing recent frames in RAM for motion detection and recording. It doesn't inherit from `io.BufferedIOBase` and uses different methods (`add_frame()`, `get_recent_frames()`).

The issue occurred because:

1. In `motion_storage.py`, there's a global variable named `frame_buffer` that's an instance of `CircularFrameBuffer`.
2. In the `write_wrapper` function, it was referring to this global variable when attempting to call `_original_write()`.
3. But the actual `_original_write` attribute was set on the `FrameBuffer` instance in the FastAPI module, not on the `CircularFrameBuffer`.

## The Fix

1. Modified the `modify_frame_buffer_write` function to accept an additional parameter `stream_buffer_instance`:

```python
def modify_frame_buffer_write(original_write_method, stream_buffer_instance=None):
    """
    Modify the FrameBuffer.write method to integrate with motion storage
    
    Args:
        original_write_method: The original write method of the FrameBuffer instance
        stream_buffer_instance: The specific FastAPI FrameBuffer instance to use
                              (needed to avoid confusion with CircularFrameBuffer)
    """
    # We store a reference to the actual stream buffer instance from the FastAPI module
    # This is different from our global frame_buffer which is a CircularFrameBuffer
    stream_buffer = stream_buffer_instance
    
    def write_wrapper(self_or_frame, *args, **kwargs):
        # ...implementation...
        
        # Now we use stream_buffer for original write
        if hasattr(self_or_frame, 'raw_frame'):
            result = instance._original_write(buf, *args[1:], **kwargs)
        else:  # PiCamera2 FileOutput._write call
            result = stream_buffer._original_write(buf)
            
        # And we use stream_buffer for accessing the raw frame
        if stream_buffer and hasattr(stream_buffer, 'raw_frame'):
            # While still using our CircularFrameBuffer (frame_buffer) for storage
            frame_buffer.add_frame(stream_buffer.raw_frame.copy(), datetime.datetime.now())
            # ...rest of implementation...
        
        return result
```

2. Updated the patching in the FastAPI module's lifespan function to explicitly pass the stream buffer instance:

```python
# Create the patched write method, passing our stream buffer instance explicitly
patched_write = motion_storage['modify_frame_buffer_write'](original_write, frame_buffer)
```

## Key Insights

1. **Module Globals**: Global variables in Python are module-scoped, so the `frame_buffer` in `motion_storage.py` and the `frame_buffer` in `fastapi_mjpeg_server_with_storage.py` are completely separate variables.

2. **Explicit References**: When working with multiple components that share similar names or purposes, it's important to keep explicit references to the objects rather than relying on globals.

3. **Clear Separation of Concerns**:
   - `FrameBuffer` is responsible for handling the camera stream data
   - `CircularFrameBuffer` is responsible for storing recent frames for motion detection

4. **Function Parameters vs. Globals**: Using function parameters to pass specific instances is more reliable than accessing global variables when there's potential for ambiguity.

## Future Improvements

For better clarity and maintainability, we could:

1. Rename the classes and variables to avoid confusion:
   - `FrameBuffer` → `CameraStreamBuffer`
   - `CircularFrameBuffer` → `MotionHistoryBuffer`

2. Refactor the code to have a clearer separation of concerns between the camera streaming module and the motion detection/storage module.

3. Create a unified buffer interface that both classes implement, making the relationship between them more explicit.