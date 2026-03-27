import torch
from dataclasses import dataclass, field
from typing import Union, List, Optional
import math

# 1. Valid test case definition
valid_test_case = {
    "inputs": torch.rand(3, 4),
    "out": None
}

# 2-4. InputSpace dataclass definition
@dataclass
class InputSpace:
    """Dataclass containing all parameters that affect output shape (except 'inputs') 
    with their discretized value ranges."""
    
    # Parameter: out
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        # Discrete values for 'out' parameter
        None,  # Default case
        # Boundary and typical tensor values (must match input shape when used)
        torch.tensor([[]]),  # Empty tensor (0-dim)
        torch.tensor(0.0),  # Scalar
        torch.zeros(1),  # 1D single element
        torch.zeros(3, 4),  # 2D matching example shape
        torch.zeros(2, 3, 4),  # 3D tensor
        torch.zeros(1, 1, 1, 1),  # 4D tensor
        # Different dtypes (all float types supported by bessel_y1)
        torch.zeros(3, 4, dtype=torch.float32),
        torch.zeros(3, 4, dtype=torch.float64),
        # Special values
        torch.full((3, 4), float('inf')),
        torch.full((3, 4), float('-inf')),
        torch.full((3, 4), float('nan')),
        # Strided tensor
        torch.zeros(6, 8)[::2, ::2],
        # Non-contiguous tensor
        torch.zeros(3, 8)[:, ::2],
        # CUDA tensor (if available)
        torch.zeros(3, 4, device='cuda') if torch.cuda.is_available() else torch.zeros(3, 4)
    ])