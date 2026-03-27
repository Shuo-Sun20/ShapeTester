import torch
from dataclasses import dataclass, field
from typing import Union, List

# 1. valid_test_case
valid_test_case = {
    'inputs': torch.tensor([-3., -2, -1, 1, 2, 3]),
    'other': torch.tensor([2.0]),
}

# 2. & 3. & 4. InputSpace with all parameters affecting output shape (except inputs)
@dataclass
class InputSpace:
    # other: shape-affecting parameter
    # Discretized value space covering:
    # - Boundary values: scalar 0, very small positive/negative
    # - Typical values: scalar positives/negatives, broadcastable tensors
    other: List[Union[float, torch.Tensor]] = field(default_factory=lambda: [
        0.0,                        # Boundary: zero
        1e-6,                       # Boundary: very small positive
        -1e-6,                      # Boundary: very small negative
        2.5,                        # Typical: positive scalar
        torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Typical: broadcastable tensor
    ])