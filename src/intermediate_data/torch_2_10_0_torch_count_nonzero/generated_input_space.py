import torch
from dataclasses import dataclass, field

# 1. Valid test case
x = torch.randn(4, 5)
valid_test_case = {
    "inputs": [x],
    "dim": 1
}

# 2, 3, 4. InputSpace definition
@dataclass
class InputSpace:
    dim: list = field(default_factory=lambda: [
        None,
        0,
        1,
        -1,
        (0, 1)
    ])