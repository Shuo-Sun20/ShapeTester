import torch
from dataclasses import dataclass, field
from typing import Optional, List

def call_func(inputs, out=None):
    return torch.log10(inputs, out=out)

example_input = torch.rand(5)
valid_test_case = {
    "inputs": example_input,
    "out": None
}

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.tensor([0.0] * 5, dtype=torch.float32),
        torch.tensor([0.0] * 5, dtype=torch.float32).view(1, 5),
        torch.tensor([0.0] * 5, dtype=torch.float32).reshape(5, 1),
        torch.tensor([0.0] * 5, dtype=torch.float64)
    ])