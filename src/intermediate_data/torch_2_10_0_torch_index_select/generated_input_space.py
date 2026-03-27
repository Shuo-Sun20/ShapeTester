import torch
from dataclasses import dataclass, field
from typing import List, Union

valid_test_case = {
    "inputs": torch.randn(3, 4),
    "dim": 0,
    "index": torch.tensor([0, 2]),
    "out": None
}

@dataclass
class InputSpace:
    dim: List[int] = field(default_factory=lambda: [-2, -1, 0, 1])
    index: List[Union[torch.Tensor, List[int]]] = field(default_factory=lambda: [
        torch.tensor([]),
        torch.tensor([0]),
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([2, 1, 0])
    ])