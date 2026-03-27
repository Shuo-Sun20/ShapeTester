import torch

def call_func(inputs, min=None, max=None, out=None):
    return torch.clamp(input=inputs, min=min, max=max, out=out)

a = torch.randn(4)
example_output = call_func(inputs=a, min=-0.5, max=0.5)