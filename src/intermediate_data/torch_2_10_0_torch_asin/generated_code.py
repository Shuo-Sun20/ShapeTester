import torch

def call_func(inputs, out=None):
    return torch.asin(inputs[0], out=out)

random_tensor = torch.rand(4) * 2 - 1  # Values in [-1, 1] for valid asin domain
example_output = call_func([random_tensor])