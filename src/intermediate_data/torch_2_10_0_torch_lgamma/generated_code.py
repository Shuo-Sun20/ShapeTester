import torch

def call_func(inputs, out=None):
    if isinstance(inputs, list):
        return torch.lgamma(inputs[0], out=out)
    else:
        return torch.lgamma(inputs, out=out)

input_tensor = torch.randn(3, 3)
example_output = call_func(input_tensor)