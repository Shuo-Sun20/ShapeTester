import torch

def call_func(inputs):
    return torch.isnan(inputs)

example_input = torch.tensor([1.0, float('nan'), 2.0, float('nan'), 3.0])
example_output = call_func(example_input)