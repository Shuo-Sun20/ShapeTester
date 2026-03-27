import torch
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Valid test case
valid_test_case = {
    "inputs": [],
    "window_length": 10,
    "periodic": True,
    "beta": 12.0,
    "dtype": None,
    "layout": torch.strided,
    "device": None,
    "requires_grad": False
}

# 2. Parameters affecting output shape: window_length
# 3. Discretized value space for window_length: 5 typical values including boundaries
@dataclass
class InputSpace:
    window_length: List[int] = field(default_factory=lambda: [1, 5, 10, 15, 20])