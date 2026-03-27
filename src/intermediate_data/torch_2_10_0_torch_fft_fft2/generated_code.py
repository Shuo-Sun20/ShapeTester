import torch

def call_func(inputs, s=None, dim=(-2, -1), norm=None, out=None):
    return torch.fft.fft2(input=inputs, s=s, dim=dim, norm=norm, out=out)

x = torch.randn(10, 10, dtype=torch.complex64)
example_output = call_func(inputs=x)