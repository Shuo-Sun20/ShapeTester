import torch

def call_func(inputs, p=None, out=None):
    return torch.linalg.cond(inputs, p, out=out)

torch.manual_seed(42)
example_input = torch.randn(3, 3)
example_output = call_func(inputs=example_input, p=2)