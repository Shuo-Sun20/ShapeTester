import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

# 1. Valid test case
example_tensor = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0])
valid_test_case = {
    "inputs": example_tensor,
    "out": None
}

# 2, 3, 4. InputSpace dataclass with parameters affecting output shape
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that can affect the output tensor shape
    (excluding 'inputs') with their discretized value ranges
    """
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    ])