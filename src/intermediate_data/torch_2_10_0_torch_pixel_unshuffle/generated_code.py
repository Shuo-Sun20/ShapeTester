import torch

def call_func(inputs, downscale_factor):
    return torch.pixel_unshuffle(inputs[0], downscale_factor)

input_tensor = torch.randn(1, 1, 12, 12)
example_output = call_func([input_tensor], 3)