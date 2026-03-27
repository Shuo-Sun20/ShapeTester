import torch
import torch.fft
from dataclasses import dataclass, field
from typing import List, Optional

torch.manual_seed(0)
input_tensor = torch.randn(5)

valid_test_case = {
    'inputs': [input_tensor],
    'n': None,
    'dim': -1,
    'norm': None,
    'out': None
}

@dataclass
class InputSpace:
    n: List[Optional[int]] = field(default_factory=lambda: [None, 1, 2, 3, 4, 5, 6, 7, 8, 16])
    dim: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2])