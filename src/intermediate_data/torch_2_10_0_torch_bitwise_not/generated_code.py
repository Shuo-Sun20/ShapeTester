import torch

def call_func(inputs, out=None):
    return torch.bitwise_not(inputs, out=out)

example_output = call_func(torch.randint(-10, 10, (5,), dtype=torch.int8))