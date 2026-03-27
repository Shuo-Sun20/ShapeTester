import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional, List

# 1. Define valid_test_case
valid_test_case = {
    'inputs': torch.randn(3, 4, 5),
    'dim': 1,
    'keepdim': True,
    'dtype': torch.int64,
    'mask': torch.randint(0, 2, (3, 4, 5), dtype=torch.bool)
}

# 2. Parameters affecting output shape: dim, keepdim
# 3. Value space analysis

@dataclass
class InputSpace:
    """
    Data class containing all parameters that affect the output tensor shape
    for torch.masked.argmin operation, with discretized value ranges.
    """
    
    # dim: int or tuple of ints
    # For 3D input tensor (3,4,5), valid dim values range from -3 to 2
    # Boundary values: -3, -2, -1, 0, 1, 2
    # Including valid_test_case value: 1
    dim: List[Union[int, Tuple[int, ...]]] = field(default_factory=lambda: [
        -3,                # negative boundary
        -2, -1,            # other negative values
        0,                 # zero
        1,                 # from valid_test_case
        2,                 # positive boundary
        (0, 1),            # multiple dimensions (tuple)
        (0, 2),            # multiple dimensions with gap
        (-3, -2),          # negative tuple
        (0, 1, 2)          # all dimensions
    ])
    
    # keepdim: bool
    # Discrete boolean values
    # Boundary/typical values: False, True
    # Including valid_test_case value: True
    keepdim: List[bool] = field(default_factory=lambda: [
        False,             # squeeze dimension(s)
        True               # keep dimension(s) as size 1
    ])

# Note: The following parameters do NOT affect output shape:
# - inputs: Tensor (excluded by requirement)
# - dtype: torch.dtype or None (affects data type, not shape)
# - mask: Tensor or None (affects computation, not output shape)

# Example instantiation
var = InputSpace()