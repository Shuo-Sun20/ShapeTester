import torch
from dataclasses import dataclass, field
from typing import Optional

def call_func(inputs, out=None):
    return torch.sgn(input=inputs, out=out)

# Valid test case
real_part = torch.randn(4)
imag_part = torch.randn(4)
example_input = torch.complex(real_part, imag_part)
valid_test_case = {
    "inputs": example_input,
    "out": None
}

@dataclass
class InputSpace:
    """Discretized value space for parameters affecting output shape."""
    out: list = field(default_factory=lambda: [
        None,
        torch.tensor([], dtype=torch.complex64),
        torch.complex(torch.tensor([0.5]), torch.tensor([-0.5])),
        torch.complex(torch.tensor([1.0, -1.0]), torch.tensor([0.0, 2.0])),
        torch.complex(torch.randn(3, 4), torch.randn(3, 4)),
        torch.complex(torch.randn(2, 3, 5), torch.randn(2, 3, 5))
    ])