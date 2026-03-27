import torch

def call_func(inputs, out=None):
    return torch.special.sinc(inputs[0], out=out)

tensor_input = torch.randn(4)
example_output = call_func([tensor_input])