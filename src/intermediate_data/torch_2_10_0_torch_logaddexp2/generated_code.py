import torch

def call_func(inputs, out=None):
    input_tensor = inputs[0]
    other_tensor = inputs[1]
    return torch.logaddexp2(input_tensor, other_tensor, out=out)

example_output = call_func([torch.randn(3, 4), torch.randn(3, 4)])