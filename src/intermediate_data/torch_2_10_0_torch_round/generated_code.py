import torch

def call_func(inputs, decimals=0, out=None):
    return torch.round(inputs, decimals=decimals, out=out)

example_input = torch.randn(3, 4, dtype=torch.float32)
example_output = call_func(example_input, decimals=2)