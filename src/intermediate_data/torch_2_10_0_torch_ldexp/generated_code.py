import torch

def call_func(inputs, out=None):
    return torch.ldexp(inputs[0], inputs[1], out=out)

input_tensor = torch.randn(3, 4)
exponent_tensor = torch.randint(-5, 5, (3, 4), dtype=torch.int32)
example_output = call_func([input_tensor, exponent_tensor])