import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union

valid_test_case = {
    'inputs': [torch.randn(4), torch.randn(4, 1)],
    'alpha': 10,
    'out': None
}

@dataclass
class InputSpace:
    alpha: List[Union[int, float]] = field(default_factory=lambda: [-2.5, -1, 0, 1, 2.5])
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])