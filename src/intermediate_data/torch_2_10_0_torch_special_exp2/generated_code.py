import torch
import torch.special

def call_func(inputs, out=None):
    return torch.special.exp2(inputs[0], out=out)

example_input = torch.randn(4, 2)
example_output = call_func([example_input])