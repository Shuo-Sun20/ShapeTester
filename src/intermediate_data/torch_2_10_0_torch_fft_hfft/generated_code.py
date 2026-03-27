import torch

def call_func(inputs, n=None, dim=-1, norm=None, out=None):
    return torch.fft.hfft(input=inputs, n=n, dim=dim, norm=norm, out=out)

n = 10
real_signal = torch.randn(n)
half_hermitian = torch.fft.rfft(real_signal)
example_output = call_func(half_hermitian, n=n)