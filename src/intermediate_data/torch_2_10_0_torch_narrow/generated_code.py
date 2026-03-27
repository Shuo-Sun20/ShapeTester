import torch

def call_func(inputs, dim, start, length):
    return torch.narrow(inputs[0], dim, start, length)

example_output = call_func(
    inputs=[torch.randn(3, 4, 5)],
    dim=1,
    start=2,
    length=2
)