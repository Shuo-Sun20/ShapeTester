import torch

def call_func(inputs, weight, bias, mean, var, eps, output_scale, output_zero_point):
    input_tensor = inputs
    return torch.quantized_batch_norm(input_tensor, weight, bias, mean, var, eps, output_scale, output_zero_point)

# Generate random input tensors
input_float = torch.rand(2, 3, 4, 4)  # NCHW format
scale = 0.1
zero_point = 128
input_quantized = torch.quantize_per_tensor(input_float, scale, zero_point, torch.quint8)

weight = torch.rand(3)
bias = torch.rand(3)
mean = torch.rand(3)
var = torch.rand(3)
eps = 1e-5
output_scale = 0.2
output_zero_point = 100

example_output = call_func(input_quantized, weight, bias, mean, var, eps, output_scale, output_zero_point)