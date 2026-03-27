import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

valid_test_case = {
    'inputs': (torch.randn(3, 5, 4), torch.randn(3, 2)),
    'out': None
}

@dataclass
class InputSpace:
    # 'out' is the only parameter of call_func (excluding 'inputs') that affects output shape
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.empty(0),  # Empty tensor
        torch.randn(3, 5, 4),  # Correct shape
        torch.randn(3, 5, 4, dtype=torch.float64),  # Different dtype
        torch.randn(3, 5, 4, dtype=torch.complex128),  # Complex dtype
        torch.randn(3, 5, 4, device='cuda') if torch.cuda.is_available() else torch.randn(3, 5, 4),  # Different device
    ])