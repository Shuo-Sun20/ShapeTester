import torch
import torch.nn as nn

def call_func(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, inputs=None):
    layer_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine, bias)
    return layer_norm(inputs)

example_input = torch.randn(20, 5, 10)
example_output = call_func(normalized_shape=10, eps=1e-05, elementwise_affine=True, bias=True, inputs=example_input)