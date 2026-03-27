import torch

def call_func(inputs, offset=0, dim1=0, dim2=1):
    return torch.diagonal(inputs[0], offset=offset, dim1=dim1, dim2=dim2)

input_tensor = torch.randn(3, 3)
example_output = call_func([input_tensor])