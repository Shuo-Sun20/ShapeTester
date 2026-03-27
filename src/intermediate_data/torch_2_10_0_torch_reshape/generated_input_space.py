import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(6)],  # Random 1D tensor with 6 elements
    "shape": (2, 3)              # Target shape for reshaping
}

# 2, 3 & 4. Define InputSpace class
@dataclass
class InputSpace:
    shape: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (6,),       # 1D tensor
        (1, 6),     # 2D tensor: 1 row, 6 columns
        (6, 1),     # 2D tensor: 6 rows, 1 column
        (2, 3),     # 2D tensor: 2 rows, 3 columns
        (3, 2)      # 2D tensor: 3 rows, 2 columns
    ])