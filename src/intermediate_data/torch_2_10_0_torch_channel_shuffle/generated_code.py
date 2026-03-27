import torch

def call_func(inputs, groups):
    return torch.channel_shuffle(inputs, groups)

example_input = torch.randn(1, 4, 2, 2)
example_output = call_func(example_input, 2)