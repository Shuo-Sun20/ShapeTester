import tensorflow as tf
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

# 1. Define valid_test_case
example_input = tf.constant([[1, 2, 3], [4, 5, 6]])
valid_test_case = {
    "inputs": example_input,
    "name": None,
    "k": 1,
    "num_rows": 4,
    "num_cols": 4,
    "padding_value": 0,
    "align": "RIGHT_LEFT"
}

# 2. Parameters that affect output shape (excluding 'inputs'):
# - k: controls which diagonals are populated
# - num_rows: specifies row dimension of output matrix
# - num_cols: specifies column dimension of output matrix
# Note: padding_value and align only affect values, not shape

# 3-4. Define InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    # Parameter 'k': Can be single integer or tuple of two integers
    # Discretized value space including boundary and typical values
    k: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        # Single integer cases (5 typical values)
        -3, -2, -1, 0, 1, 2, 3,
        # Tuple cases (various ranges)
        (-2, -1), (-1, 0), (0, 1), (1, 2),
        (-2, 0), (-1, 1), (0, 2), (-3, 3),
        # Edge cases with equal bounds (single diagonal)
        (0, 0), (1, 1), (-1, -1)
    ])
    
    # Parameter 'num_rows': Integer or None
    # Discretized value space including None, boundary, and typical values
    num_rows: List[Optional[int]] = field(default_factory=lambda: [
        None,  # Infer from other parameters
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # Boundary and typical values
        15, 20, 50  # Additional typical values
    ])
    
    # Parameter 'num_cols': Integer or None
    # Discretized value space including None, boundary, and typical values
    num_cols: List[Optional[int]] = field(default_factory=lambda: [
        None,  # Infer from other parameters
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # Boundary and typical values
        15, 20, 50  # Additional typical values
    ])

# Example instantiation
var = InputSpace()