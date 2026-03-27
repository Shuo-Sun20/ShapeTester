import torch

def call_func(inputs, eps=None, out=None):
    return torch.special.logit(inputs, eps=eps, out=out)

example_tensor = torch.rand(5)
example_output = call_func(example_tensor, eps=1e-6)