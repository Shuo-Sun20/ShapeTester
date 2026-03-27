import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

# 1. Define valid_test_case
valid_test_case = {
    'inputs': (2, 3),
    'out': None,
    'dtype': torch.float32,
    'layout': torch.strided,
    'device': None,
    'requires_grad': False,
    'pin_memory': False,
    'memory_format': torch.contiguous_format
}

# 4. Define InputSpace class
@dataclass
class InputSpace:
    # Only parameters that affect shape (except inputs): out
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.empty((1,)),
        torch.empty((2, 3)),
        torch.empty((4, 5, 6)),
        torch.empty((0, 2))  # Edge case: empty tensor
    ])