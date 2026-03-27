import torch
from dataclasses import dataclass, field
from typing import Optional, List

def call_func(inputs, dim, out=None):
    return torch.logcumsumexp(inputs, dim, out=out)

# 1. Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(5),
    "dim": 0,
    "out": None
}

# 2. Parameters affecting output shape (except inputs): dim, out

# 3. Value space analysis
# dim: int (discrete), ranges from 0 to ndim-1 of input tensor
# out: Optional[Tensor] (discrete), either None or a tensor matching expected output shape

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    dim: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None, torch.tensor([])])