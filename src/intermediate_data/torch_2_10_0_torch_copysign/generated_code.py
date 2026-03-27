import torch

def call_func(inputs, out=None):
    return torch.copysign(inputs[0], inputs[1], out=out)

a = torch.randn(5)
b = torch.randn(5)
example_output = call_func(inputs=[a, b])