import torch
from dataclasses import dataclass, field
from typing import Union, List, Optional

def call_func(inputs, out=None):
    abs_tensor, angle_tensor = inputs
    return torch.polar(abs_tensor, angle_tensor, out=out)

torch.manual_seed(42)
abs_tensor = torch.randn(3, 4).abs().float()
angle_tensor = torch.randn(3, 4).float() * torch.pi
valid_test_case = {
    'inputs': [abs_tensor, angle_tensor],
    'out': None
}

@dataclass
class InputSpace:
    # The only parameter in call_func() that can affect output shape (besides 'inputs') is 'out'
    # Since 'out' must match the shape determined by 'inputs', we can only vary it between None 
    # and a tensor of the correct shape. The value space for 'out' parameter is discretized as:
    out: List[Optional[Union[None, torch.Tensor]]] = field(
        default_factory=lambda: [
            None,
            # Boundary/total shape mismatches would cause runtime errors, so we only include 
            # valid shape-compatible tensors. These represent the possible valid states:
            # 1. None (default output)
            # 2. Complex tensor with exact same shape as inputs (float32 → complex64)
            # 3. Complex tensor with exact same shape as inputs (float64 → complex128)
            # 4. Complex tensor with broadcast-compatible shape (same as inputs)
            # 5. Complex tensor with same shape but different memory layout (if applicable)
            # We'll generate examples based on a reference shape from a dummy call
            torch.empty(3, 4, dtype=torch.complex64),
            torch.empty(3, 4, dtype=torch.complex128),
            torch.empty(1, 4, dtype=torch.complex64),  # broadcastable shape
            torch.empty(3, 1, dtype=torch.complex64),  # broadcastable shape
        ]
    )