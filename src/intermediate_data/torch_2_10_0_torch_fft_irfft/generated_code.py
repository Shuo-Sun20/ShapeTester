import torch

def call_func(inputs, n=None, dim=-1, norm=None, out=None):
    return torch.fft.irfft(input=inputs, n=n, dim=dim, norm=norm, out=out)

# Generate a valid random input tensor (half-Hermitian signal from rfft)
real_tensor = torch.randn(5)  # Real-valued input
T = torch.fft.rfft(real_tensor)  # Creates valid half-Hermitian input for irfft
example_output = call_func(inputs=T, n=5)