import torch

def call_func(inputs, alpha=1.):
    return torch.nn.functional.elu_(inputs, alpha=alpha)

example_output = call_func(torch.randn(3, 4))