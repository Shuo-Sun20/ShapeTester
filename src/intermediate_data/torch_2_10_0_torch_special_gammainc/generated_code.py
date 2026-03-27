import torch

def call_func(inputs, out=None):
    return torch.special.gammainc(inputs[0], inputs[1], out=out)

input_tensor = torch.rand(3, 2) * 5 + 0.1
other_tensor = torch.rand(3, 2) * 5 + 0.1
example_output = call_func([input_tensor, other_tensor])