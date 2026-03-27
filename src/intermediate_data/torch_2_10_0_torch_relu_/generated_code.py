import torch

def call_func(inputs):
    return torch.relu_(inputs)

example_input = torch.randn(3, 4)
example_output = call_func(example_input)