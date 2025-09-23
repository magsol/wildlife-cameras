# PiCamera2 Thread Poll Error Fix

## Problem

When running the application, a threading error would occur during initialization:

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
TypeError: modify_frame_buffer_write.<locals>.write_wrapper() missing 1 required positional argument: 'buf'
```

## Root Cause

The error occurs due to how the PiCamera2 library interfaces with our `FrameBuffer` class:

1. We implement a custom `FrameBuffer` class that inherits from `io.BufferedIOBase` to handle camera frames.
2. The `FrameBuffer.write()` method is designed to be called as an instance method, expecting `self` as the first argument, followed by the buffer to write.
3. We monkey-patch this method using our `modify_frame_buffer_write()` function which wraps the original method.
4. However, PiCamera2's `FileOutput` class doesn't call the `write` method through the instance directly, but rather takes a reference to the method and calls it with only the frame data.

This creates a method binding problem:
- When `write` is called as `frame_buffer.write(buf)`, Python automatically binds `self` to `frame_buffer`.
- But when PiCamera2 calls it as `self._fileoutput.write(frame)`, it's calling the function directly without binding, causing the `TypeError` because `self` is not provided.

## The Fix

We implemented a solution that works with both calling patterns by:

1. Storing the original write method on the frame buffer instance:
```python
# In the lifespan function:
original_write = frame_buffer.write
frame_buffer._original_write = original_write
```

2. Creating a wrapper function that detects the calling pattern and adapts accordingly:
```python
def write_wrapper(self_or_frame, *args, **kwargs):
    """
    Wrapper to handle both direct calls and calls from PiCamera2's FileOutput.
    This wrapper detects whether it's being called directly as a method or 
    indirectly through PiCamera2's FileOutput._write method and adapts accordingly.
    """
    # Detect calling pattern and adapt
    if hasattr(self_or_frame, 'raw_frame'):  # It's a direct method call
        # This is a direct call with self as first arg
        instance = self_or_frame
        buf = args[0] if args else kwargs.get('buf')
        # Call the original method through the instance's _original_write attribute
        result = instance._original_write(buf, *args[1:], **kwargs)
    else:  # It's called from PiCamera2 FileOutput._write
        # In this case, self_or_frame is actually the frame data
        buf = self_or_frame
        # Use the saved original method from the frame_buffer instance
        result = frame_buffer._original_write(buf)
    
    # Process the frame (motion detection, etc.)
    # ...
    
    return result
```

3. Replacing the frame_buffer's write method with our wrapper:
```python
patched_write = motion_storage['modify_frame_buffer_write'](original_write)
frame_buffer.write = patched_write
```

## Key Insights

1. **Method Binding**: In Python, when you call `obj.method()`, Python binds `obj` to the first parameter of `method`. However, if you store a reference to `obj.method` and then call it directly, this binding doesn't happen automatically.

2. **PiCamera2 Calling Pattern**: PiCamera2's `FileOutput` class calls our write method with just one argument (the frame data) rather than with the instance and the frame data.

3. **Original Method Access**: We need to save a reference to the original bound method to call it correctly from our wrapper.

4. **Pattern Detection**: Our wrapper detects whether it's being called directly or through PiCamera2 by checking the type of the first argument.

## Testing

The fix was tested with two different calling patterns:

1. Direct calls: `frame_buffer.write(buf)`
2. PiCamera2-style calls: `fileoutput.write(frame)`

Both patterns now work correctly, and the threading error no longer occurs.

## Future-Proofing

If the PiCamera2 API changes in the future, we may need to update our detection logic, but the current implementation is robust against the current version's calling patterns and should work reliably.

## Related Files

- `fastapi_mjpeg_server_with_storage.py`: Contains the `FrameBuffer` class and patching in the lifespan function
- `motion_storage.py`: Contains the `modify_frame_buffer_write` function that creates the wrapper