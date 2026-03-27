import torch

def call_func(inputs, generator=None, out=None):
    return torch.bernoulli(inputs, generator=generator, out=out)

example_input = torch.rand(3, 3)
example_output = call_func(example_input)