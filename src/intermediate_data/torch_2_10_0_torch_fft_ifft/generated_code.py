import torch

def call_func(inputs, n=None, dim=-1, norm=None, out=None):
    return torch.fft.ifft(inputs, n, dim, norm, out=out)

torch.manual_seed(42)
example_input = torch.randn(16, dtype=torch.complex64)
example_output = call_func(example_input)