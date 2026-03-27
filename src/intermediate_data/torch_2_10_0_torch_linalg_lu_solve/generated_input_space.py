import torch
from dataclasses import dataclass, field
from typing import Optional

# Generate example tensors from the provided code
torch.manual_seed(0)
A = torch.randn(3, 3)
LU, pivots = torch.linalg.lu_factor(A)
B = torch.randn(3, 2)

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [LU, pivots, B],
    "left": True,
    "adjoint": False,
    "out": None
}

# 2 & 3. Identify parameters affecting output shape and their value spaces
# - left: bool, affects whether we solve AX=B (left=True) or XA=B (left=False)
# - adjoint: bool, affects whether we solve with A or its adjoint
# - out: Optional[Tensor], does not affect shape, but affects output mechanism

# 4. Define InputSpace dataclass with discretized value spaces
@dataclass
class InputSpace:
    left: list = field(default_factory=lambda: [True, False])
    adjoint: list = field(default_factory=lambda: [True, False])
    # out is not included as it doesn't affect output shape
    
    # Note: The shape of the output tensor is determined by:
    # - When left=True: X.shape = (*, n, k) matching B.shape
    # - When left=False: X.shape = (*, n, k) matching B.shape
    # The actual dimensions come from B and LU in 'inputs'