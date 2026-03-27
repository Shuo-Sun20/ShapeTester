import torch

def call_func(inputs, out=None):
    input_tensor, other_tensor = inputs[0], inputs[1]
    return torch.remainder(input_tensor, other_tensor, out=out)

example_output = call_func([torch.randn(3, 4), torch.randn(3, 4)])