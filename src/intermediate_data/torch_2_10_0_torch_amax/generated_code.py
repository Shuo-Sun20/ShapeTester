import torch

def call_func(inputs, dim=None, keepdim=False, out=None):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    return torch.amax(input_tensor, dim, keepdim=keepdim, out=out)

input_tensor = torch.randn(4, 4)
example_output = call_func(inputs=input_tensor, dim=1, keepdim=False)