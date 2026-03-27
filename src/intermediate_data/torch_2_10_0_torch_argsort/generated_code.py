import torch

def call_func(inputs, dim=-1, descending=False, stable=False):
    return torch.argsort(inputs, dim=dim, descending=descending, stable=stable)

example_tensor = torch.randn(4, 4)
example_output = call_func(example_tensor, dim=1, descending=False, stable=True)