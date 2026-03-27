import torch

def call_func(inputs, dtype, reduce_range):
    return torch.quantize_per_tensor_dynamic(inputs, dtype, reduce_range)

example_output = call_func(torch.randn(4), torch.quint8, False)