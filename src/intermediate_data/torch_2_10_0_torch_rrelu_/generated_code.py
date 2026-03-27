import torch

def call_func(inputs, lower=1./8, upper=1./3, training=False):
    return torch.rrelu_(inputs, lower=lower, upper=upper, training=training)

example_input = torch.randn(3, 4)
example_output = call_func(example_input, lower=0.125, upper=0.333, training=True)