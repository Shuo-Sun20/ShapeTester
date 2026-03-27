import torch

def call_func(inputs, offset=0):
    input_tensor = inputs[0]
    return torch.diagflat(input_tensor, offset)

example_output = call_func([torch.randn(3)])