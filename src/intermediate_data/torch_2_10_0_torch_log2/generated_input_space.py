import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case
def call_func(inputs, out=None):
    return torch.log2(inputs, out=out)

torch.manual_seed(42)
input_tensor = torch.rand(5)
valid_test_case = {
    "inputs": input_tensor,
    "out": None
}

# Task 2, 3 & 4: Define InputSpace with parameters affecting output shape
@dataclass
class InputSpace:
    out: Optional[List[Optional[torch.Tensor]]] = field(
        default_factory=lambda: [
            None,
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
            torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]),
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        ]
    )