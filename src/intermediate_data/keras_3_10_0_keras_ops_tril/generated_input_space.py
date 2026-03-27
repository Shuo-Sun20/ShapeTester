import keras
from dataclasses import dataclass
from typing import List, Any

# 1. Define valid test case
example_tensor = keras.random.normal(shape=(3, 3))
valid_test_case = {"inputs": example_tensor, "k": -1}

# 2 & 3. Parameter analysis
# The only parameter in call_func besides "inputs" is "k"
# However, "k" only affects which elements are zeroed, not the output shape
# The output shape is always identical to the input shape

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # No parameters besides "inputs" affect the output shape
    # Since the problem requires InputSpace to contain all parameters that affect shape,
    # and "k" does not affect shape, we don't need to include any fields
    pass