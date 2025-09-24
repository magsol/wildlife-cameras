#!/usr/bin/python3

"""
Test script to verify the fix for the threading error in the FrameBuffer write method.
This simulates both direct calls and calls through PiCamera2's FileOutput.
"""

import io
import sys
import os
from functools import partial

# Create a simple BufferedIOBase mock
class MockBufferedIO(io.BufferedIOBase):
    def __init__(self):
        super().__init__()
        self.data = None
    
    def write(self, buf):
        self.data = buf
        return len(buf)
    
    def writable(self):
        return True
        
    def readable(self):
        return False
        
    def seekable(self):
        return False

# Create a mock patching function similar to our actual implementation
def modify_write_method(original_method):
    def write_wrapper(self_or_data, *args, **kwargs):
        print(f"In write_wrapper: self_or_data type={type(self_or_data)}")
        
        # Case 1: Direct call - self_or_data is the instance
        if isinstance(self_or_data, MockBufferedIO):
            print("Direct call detected")
            buf = args[0] if args else None
            print(f"Direct call with buf={buf}")
            result = original_method(buf)  # Original method is already bound to self
            print(f"Processed direct call: result={result}")
            return result
            
        # Case 2: FileOutput call - self_or_data is the frame data
        else:
            print("FileOutput call detected")
            buf = self_or_data
            print(f"FileOutput call with buf={buf}")
            result = fb.write(buf)  # Call through the instance to maintain binding
            print(f"Processed FileOutput call: result={result}")
            return result
    
    return write_wrapper

# Create our mock file buffer
fb = MockBufferedIO()
print("Created mock file buffer")

# Save original write method
original_write = fb.write
print("Saved original write method")

# Save reference to the buffer instance in a closure
write_method = fb.write

# Patch the write method
patched_write = modify_write_method(write_method)
fb.write = patched_write
print("Patched write method")

# Test Case 1: Direct call
print("\n--- Test Case 1: Direct call ---")
result1 = fb.write(b"test direct")
print(f"Result: {result1}")
print(f"Data: {fb.data}")

# Test Case 2: FileOutput style call
print("\n--- Test Case 2: FileOutput style call ---")
# This is how PiCamera2's FileOutput would call it
fileoutput_write = fb.write  # This loses the binding to fb
result2 = fileoutput_write(b"test fileoutput")
print(f"Result: {result2}")
print(f"Data: {fb.data}")

print("\nTest completed")