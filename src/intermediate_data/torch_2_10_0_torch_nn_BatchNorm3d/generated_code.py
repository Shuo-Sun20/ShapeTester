import torch
import torch.nn as nn

def call_func(num_features, inputs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
    bn_layer = nn.BatchNorm3d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    output = bn_layer(inputs)
    return output

example_input = torch.randn(20, 100, 35, 45, 10)
example_output = call_func(num_features=100, inputs=example_input)