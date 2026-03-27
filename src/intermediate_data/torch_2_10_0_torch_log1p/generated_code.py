import torch

def call_func(inputs, out=None):
    return torch.log1p(inputs, out=out)

example_tensor = torch.randn(5)
example_output = call_func(example_tensor)