import torch
import torch.fft

def call_func(inputs, n=None, dim=-1, norm=None, out=None):
    return torch.fft.rfft(input=inputs, n=n, dim=dim, norm=norm, out=out)

inputs = torch.randn(8)
example_output = call_func(inputs)