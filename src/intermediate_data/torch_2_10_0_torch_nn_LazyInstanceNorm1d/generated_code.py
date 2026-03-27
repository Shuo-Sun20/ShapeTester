import torch
import torch.nn as nn

def call_func(eps=1e-5, momentum=0.1, affine=False, track_running_stats=False, inputs=None):
    instance_norm = nn.LazyInstanceNorm1d(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    output = instance_norm(inputs)
    return output

example_input = torch.randn(16, 8, 32)
example_output = call_func(eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, inputs=example_input)