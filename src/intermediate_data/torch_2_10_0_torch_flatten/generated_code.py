import torch

def call_func(inputs, start_dim=0, end_dim=-1):
    return torch.flatten(inputs, start_dim, end_dim)

example_input = torch.randn(2, 3, 4)
example_output = call_func(example_input, start_dim=1, end_dim=2)