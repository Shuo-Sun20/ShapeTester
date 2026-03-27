import torch
from dataclasses import dataclass, field
from typing import Optional

# 1. Define valid_test_case
torch.manual_seed(42)
valid_test_case = {
    "inputs": [torch.randn(4)],
    "out": None
}

# 3-4. Define InputSpace with discretized value ranges
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that can affect the shape of
    torch.special.expit output (excluding "inputs").
    The only parameter is "out".
    """
    out: list[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,
            torch.empty(0),  # empty tensor
            torch.randn(1),  # 1D single element
            torch.randn(3),  # 1D small size
            torch.randn(10),  # 1D medium size
            torch.randn(100),  # 1D large size
            torch.randn(4, 5),  # 2D small
            torch.randn(3, 4, 5),  # 3D
            torch.randn(2, 3, 4, 5),  # 4D
            torch.randn(1, 1, 1),  # all dimensions size 1
            torch.randn(1000),  # boundary: large 1D
            torch.randn(100, 100),  # boundary: large 2D
            torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5]),  # specific values
            torch.tensor([]),  # empty tensor with shape (0,)
            torch.tensor([[]]),  # empty tensor with shape (1, 0)
            torch.tensor([[[]]]),  # empty tensor with shape (1, 1, 0)
        ]
    )