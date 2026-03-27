import torch
from dataclasses import dataclass, field
from typing import List, Optional

valid_test_case = {
    "inputs": torch.randn(4, 3, dtype=torch.float32),
    "n": torch.tensor(3, dtype=torch.int64),
    "out": None
}

# Value space for n (degree parameter) - discrete non-negative integers
N_VALUES = [0, 1, 2, 3, 4, 5, 10, 20, 50]

@dataclass
class InputSpace:
    n: List[int] = field(default_factory=lambda: N_VALUES)