import torch
import numpy as np
import torch
import torch.nn as nn

def call_func(num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False, inputs=None):
    instance_norm = nn.InstanceNorm3d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    return instance_norm(inputs)

# Test input that causes the defect
num_features = 3
eps = 0.0001
momentum = 0.99
affine = True
track_running_stats = True
inputs = torch.randn(4, 3, 32, 32, 32)

# Dynamic output shape
dynamic_output = call_func(num_features, eps, momentum, affine, track_running_stats, inputs)
print(f"Dynamic output shape: {list(dynamic_output.shape)}")

# Static output shape with torch.compile
compiled_func = torch.compile(call_func, dynamic=True)
static_output = compiled_func(num_features, eps, momentum, affine, track_running_stats, inputs)
print(f"Static output shape: {list(static_output.shape)}")

# Meta output shape with device='meta'
try:
    meta_inputs = torch.randn(4, 3, 32, 32, 32, device='meta')
    meta_output = call_func(num_features, eps, momentum, affine, track_running_stats, meta_inputs)
    print(f"Meta output shape: {list(meta_output.shape)}")
except Exception as e:
    print(f"Meta output shape: {str(e)}")