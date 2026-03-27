import torch

def call_func(inputs):
    return torch.positive(inputs)

example_tensor = torch.randn(5, 3)
example_output = call_func(example_tensor)