import torch

def call_func(inputs):
    return torch.view_as_real(inputs)

x = torch.randn(4, dtype=torch.cfloat)
example_output = call_func(x)