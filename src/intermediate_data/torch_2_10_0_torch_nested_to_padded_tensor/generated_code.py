import torch

def call_func(inputs, padding, output_size=None, out=None):
    return torch.nested.to_padded_tensor(inputs, padding, output_size=output_size)

nt = torch.nested.nested_tensor([torch.randn(2, 5), torch.randn(3, 4)])
example_output = call_func(nt, 0.0)