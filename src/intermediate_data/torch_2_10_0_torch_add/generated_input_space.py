import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np

valid_test_case = {
    'inputs': [torch.randn(4), torch.randn(4, 1)],
    'alpha': 10,
    'out': None
}

@dataclass
class InputSpace:
    # Parameters that affect output shape (excluding 'inputs' which contains the tensors)
    alpha: List[Union[int, float]] = field(default_factory=lambda: [-10, -2.5, -1, 0, 0.5, 1, 2.5, 10, 100])
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None, torch.randn(4, 4), torch.empty(4, 4), torch.zeros(4, 4), torch.ones(4, 4), torch.randn(3, 3, 3)])