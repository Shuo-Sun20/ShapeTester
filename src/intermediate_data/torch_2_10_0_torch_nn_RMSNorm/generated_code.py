import torch
import torch.nn as nn

def call_func(normalized_shape, eps, elementwise_affine, inputs):
    layer = nn.RMSNorm(
        normalized_shape=normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine
    )
    return layer(inputs)

normalized_shape = (2, 3)
eps = torch.finfo(torch.float32).eps
elementwise_affine = True
inputs = torch.randn(2, 2, 3)
example_output = call_func(normalized_shape, eps, elementwise_affine, inputs)