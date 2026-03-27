import torch
from dataclasses import dataclass, field
from typing import List, Callable

def foo(a: torch.Tensor, b: int) -> torch.Tensor:
    return a + b

valid_test_case = {
    "func": foo,
    "inputs": [torch.randn(3, 4)],
    "b": 2
}

@dataclass
class InputSpace:
    func: List[Callable] = field(default_factory=lambda: [foo])
    b: List[int] = field(default_factory=lambda: [-10, -1, 0, 1, 2, 3, 4, 5, 10, 100])