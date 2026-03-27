import torch

def call_func(inputs, size, stride, storage_offset=None):
    input_tensor = inputs[0]
    return torch.as_strided(input_tensor, size, stride, storage_offset)

x = torch.randn(3, 3)
example_output = call_func([x], (2, 2), (1, 2))