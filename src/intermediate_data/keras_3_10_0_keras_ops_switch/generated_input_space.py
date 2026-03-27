import keras
import keras.ops as ops
from dataclasses import dataclass, field
from typing import List, Callable, Any

# Define branch functions for testing
add_fn = lambda a, b: a + b
subtract_fn = lambda a, b: a - b
multiply_fn = lambda a, b: a * b
divide_fn = lambda a, b: a / b
sum_fn = lambda a, b: ops.sum(a) + ops.sum(b)
concatenate_fn = lambda a, b: ops.concatenate([a, b], axis=0)

# Generate random tensors
x = keras.random.uniform(shape=(2, 3))
y = keras.random.uniform(shape=(2, 3))

# Define call_func as in the original code
def call_func(index: int, branches: List[Callable], inputs: List[Any]) -> Any:
    return ops.switch(index, branches, *inputs)

# 1. Valid test case
valid_test_case = {
    "index": 0,
    "branches": [add_fn, subtract_fn],
    "inputs": [x, y]
}

# 2. & 3. & 4. InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    # Parameter: index (integer scalar)
    # Value space: discrete with boundary handling (clamped to [0, len(branches)-1])
    # Include negative, valid, and out-of-bound indices
    index: List[int] = field(default_factory=lambda: [-10, -1, 0, 1, 2, 5, 10])
    
    # Parameter: branches (sequence of callables)
    # Value space: discrete combinations of branch functions
    # Include different lengths (1 to 4 branches) and different function types
    branches: List[List[Callable]] = field(default_factory=lambda: [
        [add_fn],  # Single branch
        [add_fn, subtract_fn],  # Original example
        [add_fn, subtract_fn, multiply_fn],  # Three branches
        [add_fn, subtract_fn, multiply_fn, divide_fn],  # Four branches
        [sum_fn, concatenate_fn],  # Functions with different output shapes
        [add_fn, sum_fn, concatenate_fn],  # Mixed function types
    ])

# Example instantiation
var = InputSpace()