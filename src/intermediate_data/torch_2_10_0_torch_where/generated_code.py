import torch

def call_func(inputs, out=None):
    condition, input, other = inputs
    return torch.where(condition, input, other, out=out)

condition = torch.randn(3, 2) > 0
input_tensor = torch.randn(3, 2)
other_tensor = torch.randn(3, 2)
example_output = call_func([condition, input_tensor, other_tensor])