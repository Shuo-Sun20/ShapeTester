import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Valid test case
torch.manual_seed(42)
n, k = 3, 2
A = torch.randn(n, n, dtype=torch.float64)
A = A @ A.T + torch.eye(n, dtype=torch.float64) * 1e-3
L = torch.linalg.cholesky(A)
B = torch.randn(n, k, dtype=torch.float64)
valid_test_case = {
    'inputs': [B, L],
    'upper': False,
    'out': None
}

@dataclass
class InputSpace:
    """
    Contains all parameters of call_func() that can affect the shape of the output tensor,
    with their discretized value ranges.
    """
    upper: List[bool] = field(default_factory=lambda: [True, False])
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])