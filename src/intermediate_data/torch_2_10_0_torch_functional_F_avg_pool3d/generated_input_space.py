import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# Set seed for reproducibility
torch.manual_seed(0)

valid_test_case = {
    'inputs': torch.randn(2, 3, 8, 8, 8),
    'kernel_size': 2,
    'stride': 2,
    'padding': 0,
    'ceil_mode': False,
    'count_include_pad': True,
    'divisor_override': None
}

@dataclass
class InputSpace:
    # Parameters affecting output shape (excluding 'inputs')
    kernel_size: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [1, 2, 3, 5, (2, 3, 4), (1, 1, 1), (3, 3, 3)]
    )
    stride: List[Union[int, Tuple[int, int, int], None]] = field(
        default_factory=lambda: [None, 1, 2, 3, (1, 2, 1), (2, 2, 2), (3, 3, 3)]
    )
    padding: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [0, 1, 2, 3, (1, 1, 1), (2, 3, 2), (0, 0, 0)]
    )
    ceil_mode: List[bool] = field(
        default_factory=lambda: [True, False]
    )