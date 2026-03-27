import torch
import torch.nn as nn

def call_func(num_features, inputs, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
    instance_norm = nn.InstanceNorm2d(num_features, eps, momentum, affine, track_running_stats)
    return instance_norm(inputs)

input_tensor = torch.randn(20, 100, 35, 45)
example_output = call_func(100, input_tensor)