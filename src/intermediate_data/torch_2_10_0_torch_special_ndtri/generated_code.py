import torch

def call_func(inputs, out=None):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    
    return torch.special.ndtri(input_tensor, out=out)

example_input = torch.rand(5)
example_output = call_func([example_input])