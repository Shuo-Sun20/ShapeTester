import torch
from dataclasses import dataclass
from typing import List, Optional

# 1. Valid test case
valid_test_case = {
    "inputs": [5],
    "periodic": True,
    "dtype": None,
    "layout": torch.strided,
    "device": None,
    "requires_grad": False
}

# 3 & 4. InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    periodic: List[bool] = None
    layout: List[torch.layout] = None
    
    def __post_init__(self):
        if self.periodic is None:
            self.periodic = [True, False]
        if self.layout is None:
            self.layout = [torch.strided]