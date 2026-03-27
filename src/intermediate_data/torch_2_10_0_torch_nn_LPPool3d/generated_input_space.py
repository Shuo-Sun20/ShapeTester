import torch
from dataclasses import dataclass, field

# 1. Define valid test case
valid_test_case = {
    'norm_type': 2,
    'kernel_size': 3,
    'stride': 2,
    'ceil_mode': False,
    'inputs': torch.randn(20, 16, 50, 44, 31)
}

# 2, 3, 4. Define InputSpace dataclass with parameters affecting output shape
@dataclass
class InputSpace:
    # kernel_size: int or tuple of three ints
    kernel_size: list = field(default_factory=lambda: [
        1, 2, 3, 5, 7,  # Single int values
        (1, 1, 1), (2, 2, 2), (3, 3, 3),  # Tuple variations
        (2, 3, 4), (3, 5, 7), (1, 2, 3)   # Asymmetric tuples
    ])
    
    # stride: int or tuple of three ints or None (defaults to kernel_size)
    stride: list = field(default_factory=lambda: [
        None, 1, 2, 3, 5, 7,  # Single int values (including None)
        (1, 1, 1), (2, 2, 2), (3, 3, 3),  # Tuple variations
        (1, 2, 3), (2, 1, 2), (3, 2, 1)   # Asymmetric tuples
    ])
    
    # ceil_mode: boolean
    ceil_mode: list = field(default_factory=lambda: [True, False])