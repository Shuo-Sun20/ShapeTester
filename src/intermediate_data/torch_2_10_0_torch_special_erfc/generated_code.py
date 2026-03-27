import torch

def call_func(inputs, out=None):
    return torch.special.erfc(inputs[0], out=out)

input_tensor = torch.randn(3)
example_output = call_func([input_tensor])