import torch

def call_func(inputs, out=None):
    return torch.tan(inputs[0], out=out)

input_tensor = torch.randn(4)
example_output = call_func([input_tensor])