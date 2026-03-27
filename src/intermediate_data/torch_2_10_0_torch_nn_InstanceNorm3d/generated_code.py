import torch
import torch.nn as nn

def call_func(num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False, inputs=None):
    instance_norm = nn.InstanceNorm3d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    return instance_norm(inputs)

input_tensor = torch.randn(4, 3, 32, 32, 32)
example_output = call_func(num_features=3, inputs=input_tensor)