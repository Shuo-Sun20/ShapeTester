import keras
from dataclasses import dataclass

def call_func(inputs, dtype):
    return keras.ops.cast(x=inputs, dtype=dtype)

x = keras.random.normal(shape=(3, 4))

# Task 1: Define valid_test_case
valid_test_case = {"inputs": x, "dtype": "float16"}

# Task 4: Define InputSpace class
@dataclass
class InputSpace:
    dtype: list = None
    
    def __post_init__(self):
        if self.dtype is None:
            # Task 3: Select 5 representative dtype values
            # Common floating point types
            self.dtype = ["float16", "float32", "float64"]
            # Common integer types
            self.dtype.append("int32")
            # Boolean type
            self.dtype.append("bool")