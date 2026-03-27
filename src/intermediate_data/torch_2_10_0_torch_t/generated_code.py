import torch

def call_func(inputs):
    return torch.t(inputs)

example_input = torch.randn(2, 3)
example_output = call_func(example_input)