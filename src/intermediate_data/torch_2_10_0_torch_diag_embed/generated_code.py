import torch

def call_func(inputs, offset=0, dim1=-2, dim2=-1):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    return torch.diag_embed(input_tensor, offset=offset, dim1=dim1, dim2=dim2)

example_inputs = [torch.randn(2, 3)]
example_output = call_func(example_inputs)