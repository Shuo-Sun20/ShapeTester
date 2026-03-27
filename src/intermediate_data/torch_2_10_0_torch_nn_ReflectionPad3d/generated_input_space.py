import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple

# Valid test case ensuring API call succeeds
valid_test_case = {
    'padding': 1,
    'inputs': torch.arange(8, dtype=torch.float).reshape(1, 1, 2, 2, 2)
}

# Type alias for padding parameter
PaddingType = Union[int, Tuple[int, int, int, int, int, int]]

@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output shape with discretized value ranges."""
    
    # Parameter: padding (affects output shape)
    # Value space includes all legal values for example input shape (1,1,2,2,2)
    padding: List[PaddingType] = field(default_factory=lambda: [
        # Integer padding values (discrete)
        0,  # Minimum (no padding)
        1,  # Valid_test_case value (maximum for 2x2x2 input)
        # Tuple padding values (discrete combinations)
        (0, 0, 0, 0, 0, 0),  # No padding
        (1, 1, 1, 1, 1, 1),  # Uniform padding (1)
        (1, 0, 1, 0, 1, 0),  # Asymmetric padding
        (0, 1, 0, 1, 0, 1),  # Opposite asymmetric
        (2, 0, 0, 0, 0, 0),  # Single dimension padding
        (0, 0, 2, 0, 0, 0),  # Different dimension
        (0, 0, 0, 0, 0, 2),  # Another dimension
        # Boundary cases for tuple padding
        (0, 1, 0, 1, 0, 1),  # Mixed zeros and ones
        (1, 1, 0, 0, 1, 1),  # Patterned padding
    ])