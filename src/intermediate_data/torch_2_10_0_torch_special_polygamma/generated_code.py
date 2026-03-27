import torch

def call_func(n, inputs, out=None):
    input_tensor = inputs[0]
    return torch.special.polygamma(n, input_tensor, out=out)

example_output = call_func(1, [torch.randn(3, 4)])