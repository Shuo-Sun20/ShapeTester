import torch

def call_func(inputs, out=None):
    return torch.linalg.eigvals(inputs, out=out)

torch.manual_seed(42)
inputs = torch.randn(2, 2, dtype=torch.float64)
example_output = call_func(inputs)