import torch
from dataclasses import dataclass, field

def call_func(inputs, out=None):
    return torch.frac(input=inputs, out=out)

example_input = torch.randn(3, 4)

# 1. Define valid_test_case
valid_test_case = {
    "inputs": example_input,
    "out": None
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [
        None,
        torch.zeros(3, 4),
        torch.zeros(1, 4),
        torch.zeros(3, 1),
        torch.zeros(1, 1)
    ])