import torch

def call_func(inputs):
    return torch.real(inputs[0])

example_input = torch.randn(4, dtype=torch.cfloat)
example_output = call_func([example_input])