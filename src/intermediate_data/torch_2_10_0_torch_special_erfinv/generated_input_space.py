import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

valid_test_case = {
    'inputs': [torch.tensor([0.0, 0.5, 0.8, -0.3, -0.9])],
    'out': None
}

@dataclass
class InputSpace:
    """
    Contains all parameters of call_func that can affect the output tensor shape,
    along with their discretized value spaces.
    """
    
    # The 'out' parameter can affect the output shape when provided
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,
            torch.empty(5),           # Same shape as input
            torch.empty(3, 4),        # 2D tensor
            torch.empty(2, 3, 2),     # 3D tensor
            torch.empty(()),          # Scalar (0D)
            torch.empty(1, 5, 1),     # Broadcastable shape
            torch.empty(10),          # Larger 1D shape
            torch.empty(2, 2, 2, 2),  # 4D tensor
        ]
    )