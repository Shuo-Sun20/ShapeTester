import torch

def call_func(inputs, out=None):
    return torch.floor_divide(inputs[0], inputs[1], out=out)

input_tensor = torch.randn(3, 4) * 10
other_tensor = torch.randn(3, 4) + 0.5
example_output = call_func([input_tensor, other_tensor])