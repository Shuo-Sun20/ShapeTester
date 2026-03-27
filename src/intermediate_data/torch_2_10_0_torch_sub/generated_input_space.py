import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union

valid_test_case = {
    'inputs': [torch.randn(3, 4), torch.randn(3, 4)],
    'alpha': 2,
    'out': None
}

@dataclass
class InputSpace:
    """Input space for parameters affecting output tensor shape in torch.sub"""
    out: List[Optional[Union[torch.Tensor, None]]] = field(
        default_factory=lambda: [
            None,                                                                                # No output tensor provided
            torch.empty(3, 4),                                                                   # Correct shape, default dtype
            torch.empty(3, 4, dtype=torch.float32),                                             # Correct shape, specific dtype
            torch.empty(3, 4, dtype=torch.float64),                                             # Correct shape, different dtype
            torch.empty(3, 4, dtype=torch.complex64),                                           # Correct shape, complex dtype
            torch.empty(3, 4, requires_grad=True),                                              # Correct shape, requires gradient
            torch.empty(3, 4, device='cpu'),                                                    # Correct shape, specific device
            torch.empty(0),                                                                      # Edge case: empty tensor
            torch.empty(1, 1, 3, 4),                                                            # Edge case: different dimensions but broadcastable
            torch.empty(4),                                                                      # Invalid case for testing (won't match broadcasted shape)
            torch.empty(2, 5)                                                                    # Invalid case for testing (shape mismatch)
        ]
    )