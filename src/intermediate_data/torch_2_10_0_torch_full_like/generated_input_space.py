import torch
from dataclasses import dataclass
from typing import List, Optional

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [torch.randn(2, 3)],
    'fill_value': 5.0,
    'dtype': None,
    'layout': None,
    'device': None,
    'requires_grad': False,
    'memory_format': torch.preserve_format
}

# 2. Shape-affecting parameters (excluding "inputs"): 
# The shape is solely determined by the input tensor in "inputs"

# 3. & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    pass  # No parameters other than 'inputs' affect the shape