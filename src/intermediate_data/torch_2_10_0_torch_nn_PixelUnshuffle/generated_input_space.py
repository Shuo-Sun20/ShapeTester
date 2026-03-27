import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

def call_func(downscale_factor, inputs):
    pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
    return pixel_unshuffle(inputs)

# 1. Valid test case
valid_test_case = {
    'downscale_factor': 3,
    'inputs': torch.randn(1, 1, 12, 12)
}

# 2 & 3. Parameters affecting output shape and their value spaces
# Only parameter besides 'inputs' is 'downscale_factor'
# It must be a positive integer, and input dimensions must be divisible by it
# Value space includes boundary (1), typical positive integers, and the valid test case value

@dataclass
class InputSpace:
    downscale_factor: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10]  # Includes boundary (1) and typical values
    )