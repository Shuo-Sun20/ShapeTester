import torch
from dataclasses import dataclass, field
from typing import Union, List, Optional

# 1. Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(3, 3),
    "out": None
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameter that can affect output shape (except "inputs")
    out: List[Optional[Union[str, torch.Tensor]]] = field(
        default_factory=lambda: [
            None,  # No output tensor provided
            "zeros_like",  # Create zeros tensor with same shape as input
            "ones_like",  # Create ones tensor with same shape as input
            "empty_like",  # Create uninitialized tensor with same shape as input
            "full_like",  # Create tensor filled with value 1.0 (same shape)
            torch.tensor([1.0, 2.0, 3.0]),  # 1D tensor (requires matching input shape)
            torch.randn(2, 2),  # 2D tensor (requires matching input shape)
            torch.randn(1, 1, 1),  # 3D tensor (requires matching input shape)
        ]
    )