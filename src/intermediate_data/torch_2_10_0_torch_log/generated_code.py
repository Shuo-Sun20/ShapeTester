import torch

def call_func(inputs, out=None):
    return torch.log(input=inputs, out=out)

example_output = call_func(inputs=torch.rand(5) * 5)