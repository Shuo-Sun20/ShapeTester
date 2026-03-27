import torch

def call_func(inputs, lambd=0.5):
    return torch.functional.F.softshrink(inputs, lambd)

inputs = torch.randn(2, 4)
example_output = call_func(inputs)