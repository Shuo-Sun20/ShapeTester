import torch

def call_func(inputs, out=None):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    return torch.special.gammaln(input_tensor, out=out)

example_input = torch.rand(3, 2)
example_output = call_func(example_input)