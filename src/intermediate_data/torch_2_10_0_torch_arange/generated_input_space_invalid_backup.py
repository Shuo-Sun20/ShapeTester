import torch
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [],
    "start": 1,
    "end": 4,
    "step": 1,
    "out": None,
    "dtype": None,
    "layout": torch.strided,
    "device": None,
    "requires_grad": False
}

# 2. Parameters affecting output shape: start, end, step

# 3. & 4. InputSpace dataclass with discretized value spaces
@dataclass
class InputSpace:
    start: List[float] = field(default_factory=lambda: [0.0, 1.0, -2.5, 3.14, 10.0])
    end: List[float] = field(default_factory=lambda: [0.0, 5.0, 10.0, -3.0, 7.5])
    step: List[float] = field(default_factory=lambda: [1.0, 2.0, 0.5, -1.0, 0.1])