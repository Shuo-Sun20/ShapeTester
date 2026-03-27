import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union

valid_test_case = {
    "inputs": [torch.randn(10, 10)],
    "s": None,
    "dim": None,
    "norm": "backward",
    "out": None
}

@dataclass
class InputSpace:
    """
    Value space for parameters affecting torch.fft.ihfftn output shape.
    
    Covers:
    - s: Signal size parameters
    - dim: Dimensions to transform
    """
    
    # Parameter s: Signal size in transformed dimensions
    s: List[Optional[Union[int, Tuple[Optional[int], ...]]]] = field(
        default_factory=lambda: [
            None,  # Default: use input size
            (10,),  # 1D transform, match size
            (5,),   # 1D transform, smaller size
            (20,),  # 1D transform, larger size
            (10, 10),  # 2D transform, match sizes
            (10, 5),   # 2D transform, mixed sizes
            (20, 20),  # 2D transform, larger sizes
            (5, 20),   # 2D transform, reversed mixed
            (10, -1),  # Partial specification: only pad first dim
            (-1, 10),  # Partial specification: only pad second dim
            (-1, -1),  # No padding in any dimension
            (10, 10, 10),  # 3D transform
            (5, 5, 5),     # 3D smaller
            (10, -1, 10),  # 3D partial
        ]
    )
    
    # Parameter dim: Dimensions to transform
    dim: List[Optional[Tuple[int, ...]]] = field(
        default_factory=lambda: [
            None,           # Default: all dimensions
            (0,),           # Transform first dimension only
            (1,),           # Transform second dimension only
            (0, 1),         # Transform first two dimensions
            (1, 0),         # Transform in different order
            (-1,),          # Transform last dimension
            (-2, -1),       # Transform last two dimensions
            (0, 2),         # Skip middle dimension for 3D
            (-1, -2),       # Reverse order for last two
            (0, 1, 2),      # Transform all three dimensions (for 3D input)
        ]
    )