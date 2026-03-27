import torch

def call_func(inputs, generator=None):
    return torch.poisson(inputs, generator=generator)

rates = torch.rand(4, 4) * 5
example_output = call_func(rates)