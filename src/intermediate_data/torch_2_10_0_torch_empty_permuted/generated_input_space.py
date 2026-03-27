import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

valid_test_case = {
    "inputs": [],  # Dummy input list as required by call_func signature
    "size": (2, 3, 5, 7),
    "physical_layout": (0, 2, 3, 1),
    "dtype": torch.float32,
    "layout": torch.strided,
    "device": 'cpu',
    "requires_grad": False,
    "pin_memory": False
}

@dataclass
class InputSpace:
    # size parameter: shapes of various dimensions
    size: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (2, 3, 5, 7),               # Standard shape from documentation
        (1,),                        # 0D tensor (scalar)
        (5,),                        # 1D tensor
        (3, 4),                      # 2D tensor
        (2, 3, 4),                   # 3D tensor
        (1, 2, 3, 4, 5),             # 5D tensor
        (0, 2),                      # Zero in first dimension
        (3, 0, 4),                   # Zero in middle dimension
        (2, 3, 0),                   # Zero in last dimension
        (0, 0, 0),                   # All zeros
        (1, 1, 1),                   # All ones
        (1000, 1000),                # Large dimensions
        (1, 1000000),                # Very large second dimension
        (2**10, 2**10),              # Power of two dimensions
        (2, 3, 4, 5, 6, 7),          # Higher dimensional tensor
    ])
    
    # physical_layout parameter: permutations for various ranks
    physical_layout: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (0,),                        # For 1D
        (0, 1), (1, 0),              # For 2D
        (0, 1, 2), (0, 2, 1),        # For 3D
        (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0),
        (0, 1, 2, 3), (0, 2, 3, 1),  # For 4D (including NCHW and NHWC)
        (1, 0, 2, 3), (1, 2, 3, 0),
        (2, 0, 1, 3), (2, 3, 0, 1),
        (3, 2, 1, 0),                # Reverse order
        (0, 1, 2, 3, 4),             # For 5D
        (0, 2, 3, 4, 1),
        (4, 3, 2, 1, 0),             # Reverse order for 5D
    ])