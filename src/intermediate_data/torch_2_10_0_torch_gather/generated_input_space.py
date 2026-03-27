import torch
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Valid test case
valid_test_case = {
    'inputs': [torch.randn(3, 4), torch.randint(0, 4, (3, 4))],
    'dim': 1,
    'sparse_grad': False,
    'out': None
}

# 2 & 3. Parameters affecting output shape: 'dim', 'out' (indirectly affects via pre-allocation)
# Note: 'sparse_grad' does NOT affect output shape

@dataclass
class InputSpace:
    # dim must be within [-2, 1] for 2D tensor in the example
    dim: List[int] = field(default_factory=lambda: [0, 1, -1, -2])
    sparse_grad: List[bool] = field(default_factory=lambda: [True, False, False])