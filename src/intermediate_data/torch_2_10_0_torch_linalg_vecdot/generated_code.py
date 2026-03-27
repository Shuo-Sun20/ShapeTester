import torch

def call_func(inputs, dim=-1, out=None):
    x, y = inputs
    return torch.linalg.vecdot(x, y, dim=dim, out=out)

v1 = torch.randn(3, 2)
v2 = torch.randn(3, 2)
example_output = call_func([v1, v2])