import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case
n = 4
A = torch.randn(n, n, dtype=torch.float64)
A = A @ A.mT + torch.eye(n, dtype=torch.float64)
valid_test_case = {
    'inputs': [A],
    'upper': False,
    'out': None
}

# Task 2 & 4: Define InputSpace with all parameters affecting output shape
@dataclass
class InputSpace:
    """
    Parameters affecting output tensor shape for torch.linalg.cholesky.
    Note: Only 'out' parameter can affect output shape besides 'inputs',
    but 'out' must have same shape as expected output.
    """
    # upper: bool parameter, does not affect output shape
    upper: List[bool] = field(default_factory=lambda: [False, True])
    
    # out: Tensor or None, must match expected output shape if provided
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.empty(4, 4, dtype=torch.float64),
        torch.empty(2, 3, 3, dtype=torch.float32),
        torch.empty(1, 1, 5, 5, dtype=torch.complex64),
        torch.empty(0, 0, dtype=torch.float64),  # empty matrix
        torch.empty(10, 10, dtype=torch.double)
    ])