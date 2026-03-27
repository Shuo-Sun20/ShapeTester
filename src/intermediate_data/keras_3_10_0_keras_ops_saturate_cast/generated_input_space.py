from dataclasses import dataclass, field
import keras.ops as ops
import numpy as np

valid_test_case = {
    "inputs": np.random.uniform(-10, 300, size=(3, 4, 5)).astype("float32"),
    "dtype": "uint8"
}

# Parameter affecting output shape: only dtype (through type promotion/overflow handling, not actual shape change)
# However, the dtype parameter controls the type conversion but doesn't change tensor shape.
# The shape remains identical to input shape regardless of dtype.

@dataclass
class InputSpace:
    # dtype parameter value space - discrete values covering all common numeric types
    dtype: list = field(default_factory=lambda: [
        # Integer types (signed and unsigned)
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        # Float types
        "float16", "float32", "float64",
        "bfloat16",
        # Boolean type
        "bool",
        # Complex types (if supported)
        "complex64", "complex128"
    ])
    # Note: The actual shape of output tensor is determined solely by the input tensor shape,
    # which is not a parameter of call_func but part of the tensor itself.
    # Therefore, only dtype is included as it affects the internal casting behavior.