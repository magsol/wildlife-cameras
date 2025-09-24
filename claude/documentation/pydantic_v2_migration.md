# Pydantic V2 Migration

## Problem

The codebase was generating deprecation warnings due to using Pydantic V1 style validators:

```
wildlife-cameras/claude/fastapi_mjpeg_server_with_storage.py:289: PydanticDeprecatedSince20: Pydantic V1 style `@validator` validators are deprecated. You should migrate to Pydantic V2 style `@field_validator` validators, see the migration guide for more details. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at <https://errors.pydantic.dev/2.11/migration/>  
  @validator('rotation')  
wildlife-cameras/claude/fastapi_mjpeg_server_with_storage.py:295: PydanticDeprecatedSince20: Pydantic V1 style `@validator` validators are deprecated. You should migrate to Pydantic V2 style `@field_validator` validators, see the migration guide for more details. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at <https://errors.pydantic.dev/2.11/migration/>  
  @validator('timestamp_position')
```

## Solution

### 1. Updated Dependencies

First, we updated the Pydantic dependency to use a specific version (2.11.9):

```toml
# In pixi.toml
[dependencies]
pydantic = "==2.11.9"  # Previously was "*"
```

### 2. Updated Imports

Then, we replaced the old validator import with the new V2-style field_validator:

```python
# Old import
from pydantic import BaseModel, Field, validator

# New import
from pydantic import BaseModel, Field, field_validator
```

### 3. Updated Validator Decorators

Next, we replaced the `@validator` decorators with the new `@field_validator` decorators:

```python
# Old style
@validator('rotation')
def validate_rotation(cls, v):
    if v is not None and v not in (0, 90, 180, 270):
        raise ValueError('Rotation must be 0, 90, 180, or 270 degrees')
    return v

# New style
@field_validator('rotation')
def validate_rotation(cls, v: Optional[int]) -> Optional[int]:
    if v is not None and v not in (0, 90, 180, 270):
        raise ValueError('Rotation must be 0, 90, 180, or 270 degrees')
    return v
```

### 4. Added Type Hints

We added explicit type hints to the validator functions:

```python
# Function signature with explicit type hints
def validate_rotation(cls, v: Optional[int]) -> Optional[int]:
    # ...
```

## Key Differences Between V1 and V2 Validators

1. **Decorator Name**:
   - V1: `@validator`
   - V2: `@field_validator`

2. **Type Hints**:
   - V2 encourages more explicit type annotations
   - Added input and return type annotations to clarify expected types

3. **Validation Context**:
   - V1: Additional context via `field` and `config` parameters
   - V2: Provides a `ValidationInfo` object (though we didn't need it in our simple validators)

4. **Function Signature**:
   - V1: Less strict about parameter types
   - V2: Expects properly typed parameters and return values

## Testing

We verified that the V2-style validators work correctly by testing both valid and invalid inputs:

1. **Rotation Validation**:
   - Valid values (0, 90, 180, 270) are accepted
   - Invalid values (45, 100, 360) are rejected with appropriate errors

2. **Timestamp Position Validation**:
   - Valid values ("top-left", "top-right", "bottom-left", "bottom-right") are accepted
   - Invalid values ("center", "invalid") are rejected with appropriate errors

## Benefits

1. **No More Deprecation Warnings**: The code now uses the latest Pydantic V2 API.
2. **Future-Proof**: Ensures compatibility with future Pydantic versions.
3. **Better Type Safety**: Explicit type hints improve code clarity and help catch errors earlier.
4. **Improved Performance**: Pydantic V2 offers better performance compared to V1.

## References

- [Pydantic V2 Migration Guide](https://docs.pydantic.dev/2.11/migration/#validator-and-root_validator-are-deprecated)
- [Field Validators Documentation](https://docs.pydantic.dev/2.11/concepts/validators/#field-validators)