import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Valid test case definition
valid_test_case = {
    "inputs": torch.randn(3, 3),
    "out": None
}

# 2. Parameters affecting output shape (excluding "inputs"):
# Only "out" parameter can affect output shape when provided

# 3. Parameter type analysis and value space construction:
# out: Optional[Tensor] - Can be None or a tensor with compatible shape
# Since torch.exp requires out tensor to have same shape as input,
# we'll create value space with 5 options including boundary values

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,
            torch.empty((2, 2)),  # Same shape as input example
            torch.empty((1, 1)),  # Different smaller shape
            torch.empty((3, 3)),  # Same shape as 3x3 input
            torch.empty((2, 3)),  # Different compatible shape
        ]
    )