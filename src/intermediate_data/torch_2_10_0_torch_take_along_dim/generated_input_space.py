import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

# 1. Valid test case
torch.manual_seed(42)
valid_test_case = {
    'inputs': (torch.randn(3, 4), torch.randint(0, 4, (3, 4))),
    'dim': 1,
    'out': None
}

# 2. Parameters affecting output shape (excluding "inputs"): dim
# The output shape depends on the shape of indices tensor when dim is None,
# otherwise it's the same as indices shape.

# 3. Value space analysis:
# dim: integer or None, affects output shape
# out: doesn't affect output shape (must match expected shape if provided)

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    # dim can be integer (positive/negative) or None
    # For a typical 2D input tensor of shape (3,4):
    # - Positive values: 0 (first dim), 1 (second dim)
    # - Negative values: -1 (last dim), -2 (second-to-last dim)
    # - None: flattens input to 1D
    dim: List[Optional[int]] = None
    
    def __post_init__(self):
        # Define comprehensive value space for dim parameter
        if self.dim is None:
            # Covering all possible scenarios:
            # 1. None (flattened case)
            # 2. Positive dimensions within bounds (0, 1 for 2D tensor)
            # 3. Positive dimension at upper bound (2 for 3D tensor)
            # 4. Negative dimensions within bounds (-1, -2 for 2D tensor)
            # 5. Negative dimension at lower bound (-3 for 3D tensor)
            # 6. 0 (default)
            # Note: We include edge cases for different dimensionalities
            self.dim = [
                None,      # Flatten input
                0,         # First dimension (default)
                1,         # Second dimension (as in valid_test_case)
                -1,        # Last dimension
                -2,        # Second-to-last dimension
                2,         # Third dimension (for 3D+ tensors)
                -3         # Third-to-last dimension (for 3D+ tensors)
            ]