import torch

def call_func(inputs, s=None, dim=None, norm=None, out=None):
    input_tensor = inputs[0]
    return torch.fft.fftn(input_tensor, s=s, dim=dim, norm=norm, out=out)

x = torch.rand(10, 10, dtype=torch.complex64)
example_output = call_func(inputs=[x])