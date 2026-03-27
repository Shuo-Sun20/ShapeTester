import torch

def call_func(approximate='none', inputs=None):
    if inputs is None:
        raise ValueError("inputs must be provided")
    
    gelu_layer = torch.nn.GELU(approximate=approximate)
    output = gelu_layer(inputs)
    return output

example_output = call_func(inputs=torch.randn(2, 3))