import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# 1. Define valid_test_case dictionary
valid_test_case = {
    'inputs': torch.randn(3, 4, 5),
    'dim': 1,
    'keepdim': False,
    'dtype': None,
    'mask': torch.rand(3, 4, 5) > 0.3
}

# 2. Identify shape-affecting parameters (except "inputs")
# dim and keepdim affect the output shape

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # 3. Parameter value spaces
    
    # dim: can be int or tuple of ints
    # For 3D tensor (3,4,5): valid dims are 0,1,2
    # Negative dimensions: -1,-2,-3
    # Tuples: multiple dimensions
    dim: List[Union[int, Tuple[int, ...]]] = field(
        default_factory=lambda: [
            # Single dimensions
            0, 1, 2,  # Positive indices
            -1, -2, -3,  # Negative indices
            0,  # Boundary: first dimension
            2,  # Boundary: last dimension
            # Multiple dimensions (tuples)
            (0, 1), (0, 2), (1, 2),  # 2D reductions
            (0, 1, 2),  # All dimensions
            (-1, -2),  # Negative tuple
            (0, -1),  # Mixed positive/negative
        ]
    )
    
    # keepdim: boolean parameter
    keepdim: List[bool] = field(
        default_factory=lambda: [
            True,  # Keep reduced dimensions
            False,  # Squeeze reduced dimensions
        ]
    )