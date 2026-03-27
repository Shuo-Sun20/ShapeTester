import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Valid test case
real_tensor = torch.randn(3, 2, dtype=torch.float32)
imag_tensor = torch.randn(3, 2, dtype=torch.float32)
valid_test_case = {
    'inputs': [real_tensor, imag_tensor],
    'out': None
}

# Task 4: InputSpace definition
@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.empty((3, 2), dtype=torch.complex64),
        torch.empty((1,), dtype=torch.complex64),
        torch.empty((5, 5), dtype=torch.complex64),
        torch.empty((0, 3), dtype=torch.complex64)
    ])