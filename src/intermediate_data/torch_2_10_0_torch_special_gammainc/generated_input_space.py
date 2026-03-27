import torch
from dataclasses import dataclass, field
from typing import Optional, List

def call_func(inputs, out=None):
    return torch.special.gammainc(inputs[0], inputs[1], out=out)

# Valid test case
valid_test_case = {
    'inputs': [
        torch.rand(3, 2) * 5 + 0.1,
        torch.rand(3, 2) * 5 + 0.1
    ],
    'out': None
}

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.rand(3, 2) * 5 + 0.1,
        torch.rand(1, 2) * 5 + 0.1,
        torch.rand(2, 3) * 5 + 0.1,
        torch.tensor([])
    ])