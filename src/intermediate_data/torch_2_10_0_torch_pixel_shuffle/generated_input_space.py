import torch
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(1, 12, 4, 4),
    "upscale_factor": 2
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    upscale_factor: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])