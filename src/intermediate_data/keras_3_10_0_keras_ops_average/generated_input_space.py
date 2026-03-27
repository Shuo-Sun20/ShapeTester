import keras
import numpy as np
from dataclasses import dataclass, field

# Create random input tensors for valid_test_case
np.random.seed(42)
x_data = np.random.randn(3, 4).astype(np.float32)
weights_data = np.random.randn(4).astype(np.float32)

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [x_data, weights_data],  # List containing x and weights
    "axis": 1  # Axis along which to average
}

# 2. Parameters affecting output shape (except "inputs")
# Only "axis" parameter affects the output shape
# 3. Value space for axis parameter (discrete parameter)
# For 2D input shape (3,4), axis can be: None, 0, 1, -1, -2

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # axis can be None (average all), 0, 1, -1, or -2 for 2D input
    axis: list = field(default_factory=lambda: [None, 0, 1, -1, -2])