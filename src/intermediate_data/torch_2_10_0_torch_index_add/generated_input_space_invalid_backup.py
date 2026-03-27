import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Define valid_test_case dictionary
torch.manual_seed(42)
valid_test_case = {
    "inputs": [torch.randn(5, 3), torch.randn(3, 3)],
    "dim": 0,
    "index": torch.tensor([0, 2, 4]),
    "alpha": 1.0,
    "out": None
}

# 2. & 3. Parameters affecting output shape: dim, index
@dataclass
class InputSpace:
    dim: List[int] = field(default_factory=lambda: [-2, -1, 0, 1])
    index: List[torch.Tensor] = field(default_factory=lambda: [
        torch.tensor([0]),
        torch.tensor([0, 1]),
        torch.tensor([0, 2, 4]),
        torch.tensor([1, 3]),
        torch.tensor([4])
    ])