import torch
from dataclasses import dataclass, field
from typing import List, Tuple

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(3, 4)],
    "k": 1,
    "dims": (0, 1)
}

# Task 2, 3, 4: Define InputSpace class
@dataclass
class InputSpace:
    k: List[int] = field(default_factory=lambda: [-10, -4, -2, -1, 0, 1, 2, 4, 10])
    dims: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (0, 1),
        (1, 2),
        (0, 2),
        (2, 0),
        (1, 0),
        (2, 1),
        (0, 1, 2),
        (1, 0, 2),
        (2, 1, 0)
    ])