import torch
from dataclasses import dataclass, field
from typing import Union, Optional, List

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': [torch.tensor([1, 2, 3], dtype=torch.int32), torch.tensor([1, 3, 7], dtype=torch.int32)],
    'out': None
}

# Task 2 & 3: Parameters affecting output shape (except "inputs") and their discretized value spaces
# Parameter: out
# Type: Optional[Tensor]
# Value space: [None] (discrete)
# Note: The shape is determined by broadcasting input tensors. The 'out' parameter must match this shape
#       but doesn't affect it. Only None is valid for automatic output creation.

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Only parameter affecting output shape (except inputs) is 'out'
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])
    # Note: The list contains all possible values for 'out' parameter
    # Since 'out' must match the broadcasted shape of inputs, we only include None
    # which allows automatic output tensor creation