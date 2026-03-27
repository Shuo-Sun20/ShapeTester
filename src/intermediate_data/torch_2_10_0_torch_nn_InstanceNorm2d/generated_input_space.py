import torch
from dataclasses import dataclass, field
from typing import List, Any

# 1. Valid test case
valid_test_case = {
    'num_features': 100,
    'inputs': torch.randn(20, 100, 35, 45),
    'eps': 1e-5,
    'momentum': 0.1,
    'affine': False,
    'track_running_stats': False
}

# 2. Parameters affecting output shape (except inputs): num_features

# 3. and 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    # Only parameter affecting output shape is num_features (must match input channels)
    # Typical values covering boundary and common scenarios
    num_features: List[int] = field(default_factory=lambda: [1, 32, 64, 100, 256])