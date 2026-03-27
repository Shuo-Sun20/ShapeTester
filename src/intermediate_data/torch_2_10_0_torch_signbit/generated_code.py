import torch

def call_func(inputs, out=None):
    return torch.signbit(inputs, out=out)

example_input = torch.tensor([0.7, -1.2, 0., 2.3, -0.0, 0.0])
example_output = call_func(example_input)