from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional, Callable

# 1. Valid test case
valid_test_case = {
    "inputs": [
        torch.randn(4, 16),
        torch.randn(4, 16),
        torch.randn(4, 16)
    ],
    "distance_function": None,
    "margin": 1.0,
    "swap": False,
    "reduction": "mean"
}

# 2-4. InputSpace dataclass with parameters affecting output shape
@dataclass
class InputSpace:
    reduction: list[str] = ("none", "mean", "sum")
    
    def __post_init__(self):
        if not isinstance(self.reduction, list):
            self.reduction = list(self.reduction)

# Validation that InputSpace can be instantiated
var = InputSpace()