#!/usr/bin/python3

"""
Simple test script to demonstrate the method binding issue and its solution.
"""

import io
import types
from functools import partial

class MockFileOutput:
    def __init__(self, file_obj):
        self.file_obj = file_obj
        
    def write_frame(self, frame):
        print(f"MockFileOutput writing frame of size {len(frame)}")
        # This is how PiCamera2 calls write
        return self.file_obj.write(frame)
        
class MockBuffer(io.BufferedIOBase):
    def __init__(self):
        super().__init__()
        self.data = None
        
    def write(self, buf):
        print(f"MockBuffer.write called with buf size {len(buf)}")
        self.data = buf
        return len(buf)
        
    def writable(self):
        return True

# Create a mock of our wrapper function
def create_patched_write(buffer_instance, original_write):
    """Create a patched write function that works with both calling patterns"""
    
    def patched_write(first_arg, *args, **kwargs):
        """
        Handles both:
        1. Direct calls: buffer_instance.write(data) -> patched_write(buffer_instance, data)
        2. PiCamera2 calls: file_output.write(frame) -> patched_write(frame)
        """
        print(f"In patched_write: first_arg type={type(first_arg)}")
        
        if isinstance(first_arg, MockBuffer):
            # Case 1: Direct call as a method
            print("Direct method call")
            data = args[0] if args else None
            # Call original method directly
            print(f"Calling original_write with data={data}")
            # Original method is already bound, just pass data
            return original_write(data)
        else:
            # Case 2: Called from PiCamera2
            print("PiCamera2-style call")
            data = first_arg
            # Use the buffer instance directly 
            print(f"Calling original_write with data={data}")
            # Need to use the instance's original method
            return buffer_instance._original_write(data)
    
    return patched_write

# Solution 3: Use a descriptor to maintain binding
class MethodDescriptor:
    """A descriptor that ensures proper method binding"""
    def __init__(self, buffer_instance, original_method):
        self.buffer_instance = buffer_instance
        self.original_method = original_method
        
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return types.MethodType(self.__call__, obj)
        
    def __call__(self, obj_or_data, *args, **kwargs):
        # Is this a direct call or from FileOutput?
        if isinstance(obj_or_data, MockBuffer):
            # Direct call
            print("Descriptor: Direct method call")
            data = args[0] if args else None
            # Original method is bound to obj_or_data
            return self.original_method(data)
        else:
            # PiCamera2 call
            print("Descriptor: PiCamera2-style call")
            data = obj_or_data
            # Access the original method via stored instance
            return self.buffer_instance._original_write(data)

# Create a buffer and save its original write method
buffer = MockBuffer()
original_write = buffer.write

# METHOD 1: Simple function replacement
# This works for direct calls but fails for PiCamera2-style calls
print("\n=== METHOD 1: Simple function replacement ===")
# Save original method in instance for use by wrapper
buffer._original_write = original_write
simple_patch = create_patched_write(buffer, original_write)
buffer.write = simple_patch

print("Testing direct call...")
buffer.write(b"test direct 1")

print("\nTesting PiCamera2-style call...")
file_output = MockFileOutput(buffer)
file_output.write_frame(b"test picamera 1")

# METHOD 2: Using a descriptor
# This should work for both calls
print("\n=== METHOD 2: Using a descriptor ===")
# Reset buffer to test method 2
buffer = MockBuffer()
original_write = buffer.write
buffer._original_write = original_write
buffer.write = MethodDescriptor(buffer, original_write)

print("Testing direct call...")
buffer.write(b"test direct 2")

print("\nTesting PiCamera2-style call...")
file_output = MockFileOutput(buffer)
file_output.write_frame(b"test picamera 2")

print("\nTest completed")