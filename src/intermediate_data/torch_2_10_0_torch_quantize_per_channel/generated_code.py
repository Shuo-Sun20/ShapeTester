import torch

def call_func(inputs, scales, zero_points, axis, dtype):
    return torch.quantize_per_channel(inputs[0], scales, zero_points, axis, dtype)

input_tensor = torch.randn(3, 4)
scales = torch.randn(3)
zero_points = torch.randint(-128, 127, (3,), dtype=torch.int32)
axis = 0
dtype = torch.qint8

example_output = call_func([input_tensor], scales, zero_points, axis, dtype)