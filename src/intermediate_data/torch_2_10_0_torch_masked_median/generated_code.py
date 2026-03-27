import torch

def call_func(inputs, dim, keepdim=False, dtype=None, mask=None):
    return torch.masked.median(inputs, dim, keepdim=keepdim, dtype=dtype, mask=mask)

input_tensor = torch.randn(3, 4, 5)
mask_tensor = torch.rand(3, 4, 5) > 0.3
example_output = call_func(input_tensor, dim=1, mask=mask_tensor)