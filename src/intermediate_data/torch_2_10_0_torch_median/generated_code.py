import torch

def call_func(inputs, dim=None, keepdim=False, out=None):
    if dim is None:
        return torch.median(inputs)
    else:
        result = torch.median(inputs, dim=dim, keepdim=keepdim, out=out)
        return [result.values, result.indices]

example_input = torch.randn(4, 5)
example_output = call_func(example_input, dim=1)