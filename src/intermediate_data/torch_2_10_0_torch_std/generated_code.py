import torch

def call_func(inputs, dim=None, correction=1, keepdim=False, out=None):
    return torch.std(inputs, dim=dim, correction=correction, keepdim=keepdim, out=out)

example_tensor = torch.randn(4, 4)
example_output = call_func(inputs=example_tensor, dim=1, keepdim=True)