import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case
torch.manual_seed(42)
example_input = torch.randn(5, 5)
valid_test_case = {
    'inputs': [example_input],
    'out': None
}

# Task 2-4: Define InputSpace dataclass with discretized value spaces
@dataclass
class InputSpace:
    # Parameter: out (can be None or a tensor that affects output shape via broadcasting)
    # Type: Optional[torch.Tensor]
    # Discretized values: None, and tensors with various shapes that can be broadcasted
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,
            torch.empty((1,)),                    # Scalar-like (1D)
            torch.empty((3, 1)),                  # Column broadcastable
            torch.empty((1, 3)),                  # Row broadcastable
            torch.empty((3, 3)),                  # Same shape
            torch.empty(()),                      # Scalar (0D)
            torch.empty((5, 1, 1)),               # 3D with singleton dimensions
        ]
    )