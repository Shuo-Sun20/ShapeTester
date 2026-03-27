import torch

def call_func(inputs, out=None):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    
    return torch.acosh(input=input_tensor, out=out)

example_input = torch.randn(4).uniform_(1, 2)
example_output = call_func(example_input)