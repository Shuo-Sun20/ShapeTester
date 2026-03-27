import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

# 1. Define valid_test_case
valid_test_case = {
    "kernel_size": 3,
    "output_size": (13, 12),
    "output_ratio": None,
    "return_indices": False,
    "inputs": torch.randn(20, 16, 50, 32)
}

# 2. & 3. & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # kernel_size can be int or tuple, typical square kernels and rectangular ones
    kernel_size: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [
            1,  # minimum valid kernel
            2,  # small even kernel
            3,  # typical odd kernel (from valid_test_case)
            5,  # larger odd kernel
            7,  # larger kernel
            (2, 3),  # rectangular kernel
            (3, 2),  # rectangular kernel (transposed)
            (4, 4),  # square even kernel
            (1, 5),  # 1D pooling in width
            (5, 1)   # 1D pooling in height
        ]
    )
    
    # output_size can be None, int, or tuple
    # Note: We include None because either output_size or output_ratio must be defined
    output_size: List[Optional[Union[int, Tuple[int, int]]]] = field(
        default_factory=lambda: [
            None,  # when using output_ratio
            1,     # minimum output size
            10,    # small output
            13,    # from valid_test_case (for height)
            12,    # from valid_test_case (for width)
            (13, 12),  # from valid_test_case
            (25, 25),  # square output
            (10, 20),  # rectangular output
            (1, 50),   # extreme: 1 in height
            (50, 1)    # extreme: 1 in width
        ]
    )
    
    # output_ratio can be None, float, or tuple, must be in (0, 1)
    output_ratio: List[Optional[Union[float, Tuple[float, float]]]] = field(
        default_factory=lambda: [
            None,        # when using output_size
            0.01,        # near-minimum ratio
            0.1,         # small ratio
            0.25,        # quarter size
            0.5,         # half size (common use case)
            0.75,        # three-quarters size
            0.9,         # near-maximum ratio
            0.99,        # almost full size
            (0.5, 0.5),  # square half ratio
            (0.25, 0.75),# different ratios per dimension
            (0.1, 0.9)   # extreme difference
        ]
    )

# Example instantiation
var = InputSpace()