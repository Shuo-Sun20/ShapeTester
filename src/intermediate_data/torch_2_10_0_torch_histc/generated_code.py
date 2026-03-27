import torch

def call_func(inputs, bins=100, min=0, max=0, out=None):
    return torch.histc(inputs, bins=bins, min=min, max=max, out=out)

torch.manual_seed(42)
example_input = torch.randn(50)
example_output = call_func(example_input, bins=20, min=-2.0, max=2.0)