import torch
import torch.nn as nn
from dataclasses import dataclass, field

# 1. Define valid_test_case
valid_test_case = {
    'norm_type': 2,
    'kernel_size': 3,
    'stride': 2,
    'ceil_mode': False,
    'inputs': torch.randn(20, 16, 50)
}

# 2 & 3. Parameters affecting output shape: kernel_size, stride, ceil_mode
# Discretized value spaces:
#   - kernel_size: positive int, 1 <= kernel_size <= input_length (50)
#   - stride: positive int, 1 <= stride <= kernel_size (though >kernel_size is legal)
#   - ceil_mode: boolean

@dataclass
class InputSpace:
    kernel_size: list = field(default_factory=lambda: [1, 2, 3, 5, 10, 25, 50])
    stride: list = field(default_factory=lambda: [1, 2, 3, 5, 10, 25, 50])
    ceil_mode: list = field(default_factory=lambda: [True, False])