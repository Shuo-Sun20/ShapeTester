import torch
import torch.nn as nn

def call_func(downscale_factor, inputs):
    pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
    return pixel_unshuffle(inputs)

example_output = call_func(3, torch.randn(1, 1, 12, 12))