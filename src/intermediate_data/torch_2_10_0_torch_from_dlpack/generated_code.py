import torch

def call_func(inputs, device=None, copy=None):
    return torch.from_dlpack(inputs, device=device, copy=copy)

t = torch.randn(3, 4, 5)
example_output = call_func(t)