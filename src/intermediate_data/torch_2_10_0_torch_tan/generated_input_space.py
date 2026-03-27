import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [torch.randn(4)],
    'out': None
}

# 2. Parameters affecting output shape: Only "out" (when provided)
# 3. Value space analysis for "out":
#    Type: Optional[torch.Tensor]
#    Discrete possibilities:
#    - None (default, creates new tensor)
#    - torch.Tensor (must match input shape)

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.tensor([1.0, 2.0, 3.0, 4.0]),  # 1D, same shape as input
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # 2D
        torch.zeros(2, 3, 4),  # 3D
        torch.tensor([], dtype=torch.float32),  # empty tensor
        torch.tensor([1.0]),  # scalar as 1-element tensor
        torch.full((5,), 0.0),  # different size 1D
        torch.randn(3, 3),  # square matrix
        torch.tensor([1.0], requires_grad=True),  # with gradient
        torch.tensor([1.0], device='cpu'),  # explicit device
        torch.tensor([1.0], dtype=torch.float64),  # different dtype
    ])