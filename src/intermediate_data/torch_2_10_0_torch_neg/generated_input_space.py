import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Valid test case definition
valid_test_case = {
    "inputs": [torch.randn(5)],
    "out": None
}

# 2. & 3. & 4. InputSpace dataclass
@dataclass
class InputSpace:
    """
    Contains all parameters that can affect the output shape of torch.neg.
    The output shape is solely determined by the input tensor's shape.
    The 'out' parameter, if provided, must match the input shape but doesn't change it.
    """
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.zeros(5),
        torch.ones(5),
        torch.full((5,), 2.0),
        torch.full((5,), -1.0)
    ])