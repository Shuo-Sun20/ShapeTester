from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn

def call_func(upscale_factor, inputs):
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    output = pixel_shuffle(inputs)
    return output

valid_test_case = {
    "upscale_factor": 3,
    "inputs": torch.randn(1, 9, 4, 4)
}

@dataclass
class InputSpace:
    upscale_factor: List[int] = None
    
    def __post_init__(self):
        if self.upscale_factor is None:
            self.upscale_factor = [1, 2, 3, 4, 6, 8, 12]