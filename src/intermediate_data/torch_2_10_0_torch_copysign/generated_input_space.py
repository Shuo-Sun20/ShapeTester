import torch
from dataclasses import dataclass, field
from typing import List, Union, Optional, Any

valid_test_case = {
    "inputs": [torch.randn(5), torch.randn(5)],
    "out": None
}

@dataclass
class InputSpace:
    out: Optional[List[Union[torch.Tensor, None]]] = field(default_factory=lambda: [
        None,
        torch.empty(5),
        torch.empty((3, 4)),
        torch.empty(0),
        torch.empty((2, 3, 4))
    ])