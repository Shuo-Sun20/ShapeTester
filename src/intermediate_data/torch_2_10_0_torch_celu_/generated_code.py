import torch

def call_func(inputs, alpha=1.):
    return torch.celu_(inputs, alpha)

example_input = torch.randn(3, 4)
example_output = call_func(example_input, alpha=1.5)