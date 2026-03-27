import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List

# 1. Valid test case
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4, 5)
valid_test_case = {
    'inputs': [tensor1, tensor2],
    'out': None
}

# 2 & 3. Only 'out' parameter affects output shape (besides 'inputs')
# Parameter 'out' has type Optional[Tensor]
# Value space for 'out' (discrete, 5 values):
# - None (default)
# - Tensor with correct output shape and dtype
# - Tensor with correct shape but different dtype
# - Tensor with incorrect shape (smaller)
# - Tensor with incorrect shape (larger)

@dataclass
class InputSpace:
    # Only parameter affecting output shape (besides 'inputs') is 'out'
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,  # default
        torch.empty(10, 3, 5, dtype=torch.float32),  # correct shape, correct dtype
        torch.empty(10, 3, 5, dtype=torch.float64),  # correct shape, different dtype
        torch.empty(10, 3, 4, dtype=torch.float32),  # incorrect shape (smaller)
        torch.empty(10, 3, 6, dtype=torch.float32),  # incorrect shape (larger)
    ])