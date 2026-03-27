import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

def call_func(inputs, start_indices, shape):
    return keras.ops.slice(inputs, start_indices, shape)

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": keras.ops.convert_to_tensor(np.random.randn(5, 5)),
    "start_indices": [2, 1],
    "shape": [3, 3]
}

# Task 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Only 'shape' directly determines output tensor shape
    shape: List[List[int]] = field(default_factory=lambda: [
        # Legal shapes for 2D input (5,5) with various start_indices
        [0, 0],    # Empty slice
        [1, 1],    # Minimal non-empty slice
        [2, 2],    # Small slice
        [3, 3],    # Typical slice (from valid_test_case)
        [4, 4],    # Large slice
        [5, 5],    # Full input slice
        [2, 3],    # Rectangular slice
        [3, 2],    # Rectangular slice (transposed)
        [1, 5],    # Full-width slice
        [5, 1],    # Full-height slice
        [2, 0],    # Zero in one dimension
        [0, 2],    # Zero in other dimension
    ])