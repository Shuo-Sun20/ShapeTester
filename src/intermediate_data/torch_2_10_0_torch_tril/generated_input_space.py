import torch
from dataclasses import dataclass, field
from typing import Optional, List, Any

# 1. Define valid_test_case
b = torch.randn(4, 6)
valid_test_case = {
    'inputs': b,
    'diagonal': 1,
    'out': None
}

# 2. Parameters that can affect output shape (excluding "inputs"):
# Only the "out" parameter can affect output shape if provided

# 3. Value space analysis:
# - diagonal: int, can be any integer, discretized to [-5, -3, -1, 0, 1, 3, 5]
# - out: Optional[torch.Tensor], can be None or a tensor with same shape as input

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    diagonal: List[int] = field(default_factory=lambda: [-5, -3, -1, 0, 1, 3, 5])
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])

# Note: The 'out' parameter in InputSpace is simplified to [None] since
# providing actual tensor values would require specific input shapes.
# In practice, if 'out' is provided, it must be a tensor with the same
# shape as the input tensor.