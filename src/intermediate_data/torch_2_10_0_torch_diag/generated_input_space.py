import torch
from dataclasses import dataclass, field
from typing import List, Optional

valid_test_case = {
    'inputs': [torch.tensor([1.0, 2.0, 3.0])],
    'diagonal': 0,
    'out': None
}

@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the shape of the output tensor
    from torch.diag, with discretized value ranges.
    """
    diagonal: List[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])