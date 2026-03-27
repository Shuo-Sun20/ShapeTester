import torch

def call_func(inputs, nan=0.0, posinf=None, neginf=None, out=None):
    input_tensor = inputs[0]
    return torch.nan_to_num(input_tensor, nan=nan, posinf=posinf, neginf=neginf, out=out)

torch.manual_seed(42)
x = torch.randn(3, 3)
x[0, 0] = float('nan')
x[1, 1] = float('inf')
x[2, 2] = -float('inf')

example_output = call_func([x])