import torch

def call_func(inputs, dim=None):
    return torch.fft.fftshift(inputs, dim)

example_output = call_func(torch.randn(4, 4), dim=(0, 1))