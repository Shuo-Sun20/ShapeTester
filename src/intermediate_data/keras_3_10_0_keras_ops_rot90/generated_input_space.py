import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# Generate the random tensor as in the example
random_tensor = np.random.rand(3, 4, 5).astype(np.float32)

# 1. Define valid_test_case
valid_test_case = {
    "inputs": random_tensor,
    "k": 2,
    "axes": (1, 2)
}

# 2 & 3: Parameters affecting output shape (k and axes)
# k: integer, can be any integer (rotation count modulo 4)
# axes: tuple of two distinct integers within the tensor's dimensions

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    k: List[int] = None  # rotation count
    axes: List[Tuple[int, int]] = None  # rotation plane axes
    
    def __post_init__(self):
        if self.k is None:
            # Discretized k values: negative, zero, positive, covering modulo 4 cases
            self.k = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        if self.axes is None:
            # Typical axes pairs for 3D/4D tensors (covering common cases)
            self.axes = [
                (0, 1), (1, 0),  # Standard 2D rotation plane
                (0, 2), (2, 0),  # Different plane for 3D+
                (1, 2), (2, 1),  # Another plane for 3D+
                (0, 3), (3, 0),  # Extended for higher dimensions
                (2, 3), (3, 2)   # More 4D examples
            ]