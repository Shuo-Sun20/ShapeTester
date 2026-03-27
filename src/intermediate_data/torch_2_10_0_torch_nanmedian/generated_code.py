import torch

def call_func(inputs, dim=None, keepdim=False, out=None):
    if dim is None:
        return torch.nanmedian(inputs)
    else:
        return torch.nanmedian(inputs, dim=dim, keepdim=keepdim, out=out).values

example_tensor = torch.tensor([[2.0, 3.0, float('nan'), 1.0],
                               [float('nan'), 5.0, 4.0, float('nan')]])
example_output = call_func(example_tensor, dim=1, keepdim=False)