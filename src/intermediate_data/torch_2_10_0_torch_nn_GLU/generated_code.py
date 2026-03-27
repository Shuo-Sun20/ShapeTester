import torch

def call_func(inputs, dim=-1):
    glu_layer = torch.nn.GLU(dim=dim)
    return glu_layer(inputs)

example_input = torch.randn(4, 2)
example_output = call_func(example_input)