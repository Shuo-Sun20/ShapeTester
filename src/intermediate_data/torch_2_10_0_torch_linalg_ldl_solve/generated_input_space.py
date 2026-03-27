import torch
from dataclasses import dataclass
from typing import Optional, List, Any

# Task 1: Define valid_test_case
A = torch.randn(3, 3, dtype=torch.float64)
A = A @ A.T  # Make symmetric
LD, pivots, info = torch.linalg.ldl_factor_ex(A)
B = torch.randn(3, 2, dtype=torch.float64)

valid_test_case = {
    "inputs": [LD, pivots, B],
    "hermitian": False,
    "out": None
}

# Task 2: Identify parameters that affect output shape (excluding "inputs")
# The parameters are: hermitian, out. However, only 'out' can affect the shape
# when provided (in-place operation), but the output shape is determined by LD and B.
# Since the task specifies parameters in call_func (excluding "inputs"), 
# we analyze: hermitian (bool) and out (Optional[torch.Tensor]).
# The output shape is always (*, n, k) from inputs (LD: (*, n, n), B: (*, n, k)),
# so these parameters don't change the output shape.

# Task 3: Value spaces for parameters affecting shape (only 'out' in this context)
# Since 'hermitian' is boolean and doesn't affect shape, we list its possible values.
# 'out' can be None or a tensor with appropriate shape.

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters from call_func that could affect output shape (excluding 'inputs')
    hermitian: List[bool] = None  # Discrete boolean values
    out: List[Optional[torch.Tensor]] = None  # Discrete: None or tensor

    def __post_init__(self):
        if self.hermitian is None:
            self.hermitian = [False, True]  # All possible boolean values
        if self.out is None:
            # Create example tensors with valid shapes (same as example output)
            A = torch.randn(3, 3, dtype=torch.float64)
            A = A @ A.T
            LD, pivots, info = torch.linalg.ldl_factor_ex(A)
            B = torch.randn(3, 2, dtype=torch.float64)
            example_output = torch.linalg.ldl_solve(LD, pivots, B)
            # Discretized values: None, and tensors with valid shapes
            self.out = [None, example_output.clone(), torch.zeros_like(example_output)]