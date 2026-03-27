import torch

def call_func(inputs, out=None):
    return torch.isneginf(inputs, out=out)

input_tensor = torch.tensor([-float('inf'), float('inf'), torch.randn(1).item()])
example_output = call_func(input_tensor)