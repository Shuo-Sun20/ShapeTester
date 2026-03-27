import torch
from dataclasses import dataclass, field
from typing import Union, List, Optional
import numpy as np

# 1. valid_test_case definition
condition = torch.randn(3, 2) > 0
input_tensor = torch.randn(3, 2)
other_tensor = torch.randn(3, 2)
valid_test_case = {
    'inputs': [condition, input_tensor, other_tensor],
    'out': None
}

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    """Dataclass containing all parameters that affect the output shape of torch.where."""
    
    out: List[Optional[Union[torch.Tensor, None]]] = field(
        default_factory=lambda: [
            None,
            torch.tensor([], dtype=torch.float32),
            torch.zeros((1,), dtype=torch.float32),
            torch.zeros((3, 2), dtype=torch.float32),
            torch.zeros((1, 2), dtype=torch.float32),
            torch.zeros((3, 1), dtype=torch.float32),
            torch.zeros((), dtype=torch.float32)  # scalar tensor
        ]
    )