import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [
        torch.randn(5, 2, dtype=torch.float64),
        torch.randn(2, dtype=torch.float64),
        torch.randn(5, 3, dtype=torch.float64)
    ],
    'left': True,
    'transpose': False,
    'out': None
}

# 2. Parameters affecting output shape (except "inputs"): left, transpose
# 3. Value space analysis:
#    left: bool -> [True, False]
#    transpose: bool -> [True, False]

@dataclass
class InputSpace:
    left: List[bool] = field(default_factory=lambda: [True, False])
    transpose: List[bool] = field(default_factory=lambda: [True, False])