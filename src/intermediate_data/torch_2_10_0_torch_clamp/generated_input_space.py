import torch
from dataclasses import dataclass, field
from typing import Union, Optional, List

# 1. Valid test case
valid_test_case = {
    "inputs": torch.randn(4),
    "min": -0.5,
    "max": 0.5,
    "out": None
}

# 2. Parameters affecting output shape: min, max, out
# 3. Discretized value spaces

@dataclass
class InputSpace:
    # min parameter value space
    min: List[Optional[Union[float, torch.Tensor]]] = field(
        default_factory=lambda: [
            None,                         # No lower bound
            -1.0,                         # Negative boundary
            -0.5,                         # Included from test case
            0.0,                          # Zero value
            0.5,                          # Positive value
            1.0,                          # Positive boundary
            torch.tensor(-0.5),           # Scalar tensor
            torch.tensor([-0.5]),         # 1D tensor, size=1
            torch.tensor([-1.0, 0.0, 1.0, 0.5]),  # 1D tensor, size=4 (broadcastable)
            torch.tensor([[-0.5], [0.0], [0.5], [1.0]])  # 2D tensor, broadcastable
        ]
    )
    
    # max parameter value space
    max: List[Optional[Union[float, torch.Tensor]]] = field(
        default_factory=lambda: [
            None,                         # No upper bound
            -1.0,                         # Negative boundary
            -0.5,                         # Negative value
            0.0,                          # Zero value
            0.5,                          # Included from test case
            1.0,                          # Positive boundary
            torch.tensor(0.5),            # Scalar tensor
            torch.tensor([0.5]),          # 1D tensor, size=1
            torch.tensor([-0.5, 0.0, 0.5, 1.0]),  # 1D tensor, size=4 (broadcastable)
            torch.tensor([[0.5], [1.0], [1.5], [2.0]])  # 2D tensor, broadcastable
        ]
    )
    
    # out parameter value space
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,                         # No output tensor provided
            torch.empty(4),               # Same shape as input (4,)
            torch.empty(2, 4),            # Different shape (2, 4) - broadcastable
            torch.empty(4, 1),            # Different shape (4, 1) - broadcastable
            torch.empty(1, 4),            # Different shape (1, 4) - broadcastable
        ]
    )