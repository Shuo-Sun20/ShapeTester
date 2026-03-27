import torch

def call_func(inputs, dim=None, keepdim=False, dtype=None, out=None):
    return torch.nanmean(input=inputs, dim=dim, keepdim=keepdim, dtype=dtype, out=out)

example_input = torch.tensor([[torch.nan, 1.0, 2.0], [1.0, 2.0, 3.0]])
example_output = call_func(inputs=example_input, dim=0, keepdim=False)