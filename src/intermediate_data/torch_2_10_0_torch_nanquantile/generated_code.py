import torch

def call_func(inputs, q, dim=None, keepdim=False, interpolation='linear', out=None):
    return torch.nanquantile(inputs, q, dim, keepdim, interpolation=interpolation, out=out)

example_tensor = torch.tensor([[1.0, 2.0, float('nan')], [4.0, float('nan'), 6.0]])
example_q = 0.5
example_output = call_func(example_tensor, example_q, dim=1, keepdim=False, interpolation='linear')