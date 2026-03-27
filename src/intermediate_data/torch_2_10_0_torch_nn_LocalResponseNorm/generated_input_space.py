import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

def call_func(size, alpha, beta, k, inputs):
    lrn = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
    return lrn(inputs)

valid_test_case = {
    'size': 2,
    'alpha': 0.0001,
    'beta': 0.75,
    'k': 1,
    'inputs': torch.randn(32, 5, 24, 24)
}

@dataclass
class InputSpace:
    size: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    alpha: List[float] = field(default_factory=lambda: [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0])
    beta: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 2.0])
    k: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 5.0, 10.0])