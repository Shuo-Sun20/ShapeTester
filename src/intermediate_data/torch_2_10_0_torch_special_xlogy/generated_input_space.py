import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np

def call_func(inputs, out=None):
    return torch.special.xlogy(inputs[0], inputs[1], out=out)

valid_test_case = {
    "inputs": [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([3.0, 2.0, 1.0])
    ],
    "out": None
}

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,
            torch.zeros(1),
            torch.zeros(3, 4),
            torch.zeros(2, 3, 4),
            torch.empty(0),
            torch.zeros(1, 2, 3, 4),
            torch.full((3, 4), float('nan')),
            torch.full((3, 4), float('inf')),
            torch.full((3, 4), float('-inf'))
        ]
    )