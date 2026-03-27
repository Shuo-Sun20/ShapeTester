import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

valid_test_case = {
    "inputs": [],
    "size": (2, 3, 5, 7),
    "physical_layout": (0, 2, 3, 1),
    "dtype": torch.float32,
    "layout": torch.strided,
    "device": 'cpu',
    "requires_grad": False,
    "pin_memory": False
}

@dataclass
class InputSpace:
    size: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (1,), (0, 2), (2, 3, 4), (1, 5, 1, 5), (3, 0, 2, 4, 1)
    ])