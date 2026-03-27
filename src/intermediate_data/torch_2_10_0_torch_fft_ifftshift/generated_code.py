import torch

def call_func(inputs, dim=None):
    return torch.fft.ifftshift(inputs, dim)

example_output = call_func(torch.randn(10, 10))