import torch

def call_func(inputs, shape):
    return torch.broadcast_to(inputs, shape)

example_output = call_func(
    inputs=torch.randn(1, 4),
    shape=(3, 4)
)