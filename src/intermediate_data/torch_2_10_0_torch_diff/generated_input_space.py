import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

# Valid test case dictionary
valid_test_case = {
    "inputs": torch.tensor([1.0, 2.0, 3.0, 4.0]),  # Main input tensor
    "n": 1,
    "dim": -1,
    "prepend": None,
    "append": None,
    "out": None
}

@dataclass
class InputSpace:
    # n must be >= 0. We include boundary values and typical values.
    n: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    
    # dim values for 1D tensor. We include boundary values (-1, 0) and other possible values
    # For 1D tensor, only dim=-1 or dim=0 are valid
    dim: List[int] = field(default_factory=lambda: [-1, 0])
    
    # prepend can be None or a tensor that affects output shape
    # We include None and some tensor examples
    prepend: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.tensor([0.5]),
        torch.tensor([0.5, 0.8]),
        torch.tensor([]),  # Empty tensor
        torch.tensor([0.1, 0.2, 0.3])
    ])
    
    # append can be None or a tensor that affects output shape
    # We include None and some tensor examples
    append: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.tensor([4.5]),
        torch.tensor([4.5, 4.8]),
        torch.tensor([]),  # Empty tensor
        torch.tensor([4.5, 4.8, 5.0])
    ])