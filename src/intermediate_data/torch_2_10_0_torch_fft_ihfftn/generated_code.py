import torch

def call_func(inputs, s=None, dim=None, norm=None, out=None):
    return torch.fft.ihfftn(inputs[0], s=s, dim=dim, norm=norm, out=out)

example_output = call_func(
    inputs=[torch.randn(10, 10)],
    s=None,
    dim=None,
    norm="backward",
    out=None
)