import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

def call_func(inputs, out=None):
    return torch.linalg.inv(A=inputs, out=out)

# 1. Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(4, 4) @ torch.randn(4, 4).T + torch.eye(4) * 1e-3,
    "out": None
}

# 2. Parameters affecting output shape (excluding "inputs"): "out"
#    Note: The shape of output tensor is solely determined by the shape of inputs.
#    The "out" parameter does not change output shape, but must have matching shape.

# 3. Parameter types and value spaces:
#    "out": Can be None or a tensor matching input shape

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    out: List[Optional[Union[torch.Tensor, None]]] = field(
        default_factory=lambda: [
            None,  # Default case
            torch.randn(4, 4) @ torch.randn(4, 4).T + torch.eye(4) * 1e-3,  # Matching tensor
            torch.empty(4, 4),  # Empty tensor
            torch.zeros(4, 4),  # Zero tensor
            torch.ones(4, 4),   # Ones tensor
            torch.eye(4)        # Identity tensor
        ]
    )