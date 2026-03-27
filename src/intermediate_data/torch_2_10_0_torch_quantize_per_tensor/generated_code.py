import torch

def call_func(inputs, scale, zero_point, dtype):
    if isinstance(inputs, list):
        return torch.quantize_per_tensor(inputs, scale, zero_point, dtype)
    else:
        return torch.quantize_per_tensor(inputs, scale, zero_point, dtype)

input_tensor = torch.randn(4) * 2.0
scale_val = 0.1
zero_point_val = 10
dtype_val = torch.quint8

example_output = call_func(input_tensor, scale_val, zero_point_val, dtype_val)