import torch
import torch.masked

def call_func(inputs, ord, dim, eps=1e-12, dtype=None, mask=None):
    return torch.masked.normalize(inputs, ord, dim, eps=eps, dtype=dtype, mask=mask)

torch.manual_seed(42)
inputs = torch.randn(2, 3, 4)
mask = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool)
example_output = call_func(inputs, 2.0, 1, mask=mask)