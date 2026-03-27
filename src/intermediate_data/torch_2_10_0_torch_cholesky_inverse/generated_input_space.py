import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Define valid_test_case dictionary
torch.manual_seed(42)
A = torch.randn(4, 4)
A = A @ A.T + torch.eye(4) * 1e-3
L = torch.linalg.cholesky(A)
valid_test_case = {
    'inputs': [L],
    'upper': False,
    'out': None
}

# 2 & 3 & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameter 'upper' is a boolean flag - discrete parameter
    upper: List[bool] = field(default_factory=lambda: [False, True])
    
    # Parameter 'out' is optional - discretized with None and 4 tensor options
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.empty((4, 4), dtype=torch.float32),
        torch.empty((4, 4), dtype=torch.float64),
        torch.empty((4, 4), dtype=torch.complex64),
        torch.empty((4, 4), dtype=torch.complex128)
    ])