import torch

def call_func(inputs):
    return torch.flipud(inputs)

example_tensor = torch.randn(3, 4)
example_output = call_func(example_tensor)