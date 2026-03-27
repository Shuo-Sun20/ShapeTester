import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Union

# 1. Define valid_test_case
valid_test_case = {
    "eps": 1e-5,
    "momentum": 0.1,
    "affine": False,
    "track_running_stats": False,
    "device": None,
    "dtype": None,
    "inputs": [torch.randn(2, 6, 10, 8, 4)]
}

# 2. Parameters affecting output shape (excluding inputs): NONE
#    Only input tensor shape affects output shape, other parameters only affect numerical values.

# 3. Parameter analysis and value spaces (for completeness, though they don't affect shape):
#    - eps: continuous float, affects numerical stability but not shape
#    - momentum: continuous float [0,1], affects running statistics but not shape
#    - affine: discrete bool, affects learnable parameters but not shape
#    - track_running_stats: discrete bool, affects statistics tracking but not shape
#    - device: discrete torch.device/str/None, affects tensor device but not shape
#    - dtype: discrete torch.dtype/None, affects tensor dtype but not shape

# 4. InputSpace dataclass (empty since no shape-affecting parameters besides inputs)
@dataclass
class InputSpace:
    # No fields needed as no parameters affect output shape (excluding inputs)
    pass

# Example instantiation
var = InputSpace()