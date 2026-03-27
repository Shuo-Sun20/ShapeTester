import torch
import numpy as np
import torch

def call_func(
    num_features: int,
    inputs: torch.Tensor,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = False,
    track_running_stats: bool = False
) -> torch.Tensor:
    instance_norm_layer = torch.nn.InstanceNorm1d(
        num_features=num_features,
        eps=eps,
        momentum=momentum,
        affine=affine,
        track_running_stats=track_running_stats
    )
    output = instance_norm_layer(inputs)
    return output

# Test input that causes the defect
num_features = 2
inputs = torch.randn(20, 2, 40)
eps = 0.01
momentum = 0.0
affine = False
track_running_stats = True

# Get dynamic output shape
dynamic_output = call_func(num_features, inputs, eps, momentum, affine, track_running_stats)
dynamic_shape = list(dynamic_output.shape)
print(f"Dynamic output shape: {dynamic_shape}")

# Get static output shape using torch.compile
compiled_func = torch.compile(call_func, dynamic=True)
static_output = compiled_func(num_features, inputs, eps, momentum, affine, track_running_stats)
static_shape = list(static_output.shape)
print(f"Static output shape: {static_shape}")

# Get meta output shape using device='meta'
try:
    meta_inputs = inputs.to(device='meta')
    meta_output = call_func(num_features, meta_inputs, eps, momentum, affine, track_running_stats)
    meta_shape = list(meta_output.shape)
    print(f"Meta output shape: {meta_shape}")
except Exception as e:
    print(f"Meta output shape: {e}")

# Verify the defect - check for inconsistencies
print(f"\nDefect verification:")
print(f"Dynamic == Static: {dynamic_shape == static_shape}")
print(f"All shapes consistent: {dynamic_shape == static_shape}")