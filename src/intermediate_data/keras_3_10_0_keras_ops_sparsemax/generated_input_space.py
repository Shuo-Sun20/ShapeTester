import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": np.array([[-1., 0., 1.], [2., -1., 0.5]], dtype=np.float32),
    "axis": -1
}

# Task 2: Parameters that affect output shape (except inputs)
# Only the "axis" parameter affects the shape of the output tensor.

# Task 3 & 4: Define InputSpace dataclass with discretized value spaces
@dataclass
class InputSpace:
    # The axis parameter can be:
    # 1. Positive integers (0, 1, 2, ... up to input_rank-1)
    # 2. Negative integers (-1, -2, ... down to -input_rank)
    # 
    # For testing, we consider a typical input with rank 4 (e.g., batch, height, width, channels).
    # We'll discretize the value space to cover common use cases:
    # - Boundary values: -4 (first axis), -1 (last axis), 0 (first axis positive), 3 (last axis positive)
    # - Typical values: -2, -1, 0, 1, 2, 3
    
    axis: List[int] = field(
        default_factory=lambda: [
            -4, -3, -2, -1,  # Negative axis values
            0, 1, 2, 3       # Positive axis values
        ]
    )

# Example instantiation
var = InputSpace()