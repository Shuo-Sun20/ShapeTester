import torch
from dataclasses import dataclass, field
from typing import Optional, List

valid_test_case = {
    "inputs": [torch.randn(3), torch.tensor([0, 1, 2])],
    "out": None
}

@dataclass
class InputSpace:
    """Space of input parameters that affect output shape"""
    
    # Parameter: out
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.zeros(3),
        torch.ones(3),
        torch.empty(3),
        torch.randn(3),
        torch.tensor([1.0, 2.0, 3.0])
    ])