import torch
import torch.nn as nn

def call_func(eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None, inputs=None):
    norm_layer = nn.LazyInstanceNorm2d(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)
    output = norm_layer(inputs)
    return output

input_tensor = torch.randn(2, 3, 4, 5)
example_output = call_func(inputs=input_tensor)