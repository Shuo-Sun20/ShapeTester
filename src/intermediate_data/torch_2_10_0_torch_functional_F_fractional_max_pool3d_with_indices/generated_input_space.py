import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

# Task 1: Define valid_test_case
input_tensor = torch.randn(20, 16, 50, 32, 16)
valid_test_case = {
    'inputs': [input_tensor],
    'kernel_size': 3,
    'output_size': (13, 12, 11),
    'output_ratio': None,
    '_random_samples': None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters that affect output shape (excluding "inputs")
    output_size: List[Optional[Union[int, Tuple[int, int, int]]]] = field(
        default_factory=lambda: [
            None,
            10,
            15,
            (5, 5, 5),
            (8, 8, 8),
            (10, 10, 10),
            (13, 12, 11),
            (20, 16, 16),
            (25, 25, 25)
        ]
    )
    output_ratio: List[Optional[Union[float, Tuple[float, float, float]]]] = field(
        default_factory=lambda: [
            None,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.999,
            (0.1, 0.1, 0.1),
            (0.25, 0.25, 0.25),
            (0.5, 0.5, 0.5),
            (0.75, 0.75, 0.75),
            (0.9, 0.8, 0.7),
            (0.999, 0.999, 0.999)
        ]
    )